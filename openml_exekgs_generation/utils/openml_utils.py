# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import time
import openml
import pandas as pd
import requests
from openml_exekgs_generation.config import (
    DATASETS_PATH,
    FLOWS_PATH,
    METRIC_FOR_SORTING_PIPELINES_DICT,
    OPENML_DATASET_KWARGS,
    OPENML_DATATYPE_TO_EXEKG_SEMANTICS,
    OPENML_EST_PROCEDURE_PARAMS_TO_SKLEARN,
    OPENML_EST_PROCEDURE_TYPE_TO_SKLEARN,
    TASKS_PATH,
)
from openml_exekgs_generation.utils.string_utils import (
    code_param_to_exekg_param,
    openml_param_value_to_exekg_param_value,
)
from exe_kg_lib.classes.data_entity import DataEntity
from exe_kg_lib.classes.entity import Entity
from exe_kg_lib.config import KG_SCHEMAS

INITIAL_REQ_DELAY = 1  # Initial delay in seconds
MAX_REQ_DELAY = 60  # Maximum delay in seconds


def get_param_dict_from_est_procedure_params(est_procedure_params_dict):
    converted_param_dict = {}
    for param_name, param_value in est_procedure_params_dict.items():
        if not param_value:
            continue

        param_name = code_param_to_exekg_param(
            OPENML_EST_PROCEDURE_PARAMS_TO_SKLEARN[param_name]
        )
        param_value = openml_param_value_to_exekg_param_value(param_value)
        if param_value is None:
            continue

        converted_param_dict[param_name] = param_value

    return converted_param_dict


def get_est_procedure_from_task(task):
    est_procedure_type = task.estimation_procedure["type"]
    est_procedure_params_dict = task.estimation_procedure["parameters"]

    stratified_sampling = est_procedure_params_dict.pop("stratified_sampling", False)
    number_repeats = est_procedure_params_dict.get("number_repeats", None)

    if est_procedure_type not in OPENML_EST_PROCEDURE_TYPE_TO_SKLEARN:
        raise ValueError(f"Estimation procedure {est_procedure_type} not supported.")

    est_procedure_sklearn = OPENML_EST_PROCEDURE_TYPE_TO_SKLEARN[est_procedure_type]
    if stratified_sampling:
        est_procedure_sklearn = f"Stratified{est_procedure_sklearn}"

    if number_repeats:
        if int(number_repeats) > 1:
            est_procedure_sklearn = f"Repeated{est_procedure_sklearn}"
        else:
            est_procedure_params_dict.pop("number_repeats")

    est_procedure_params_dict_sklearn = get_param_dict_from_est_procedure_params(
        est_procedure_params_dict
    )

    return est_procedure_sklearn, est_procedure_params_dict_sklearn


def get_best_runs_for_task(
    task_id,
    metric,
    filter_flow_ids,
    num_runs_per_flow=None,
):
    if metric == "f_measure":
        sort_order = "desc"
    elif metric == "root_mean_squared_error":
        sort_order = "asc"
    else:
        raise ValueError(f"Metric {metric} not supported")

    evaluations_df = openml_call_with_retry(
        openml.evaluations.list_evaluations,
        **dict(
            function=metric,
            tasks=[task_id],
            output_format="dataframe",
            sort_order=sort_order,
        ),
    )
    if evaluations_df.empty:
        return None, None

    evaluations_df = evaluations_df[
        evaluations_df["flow_id"].astype(int).isin(filter_flow_ids)
    ]

    if evaluations_df.empty:
        return None, None

    if num_runs_per_flow:
        evaluations_df = (
            evaluations_df.groupby("flow_id").head(num_runs_per_flow).reset_index()
        )

    run_ids = evaluations_df["run_id"].astype(int).tolist()
    metric_values = evaluations_df["value"].tolist()

    return run_ids, metric_values


def get_flows(filter_fullname_regex=None):
    print(f"Fetching flows filtered by regex: {filter_fullname_regex}...")
    if not FLOWS_PATH.exists():
        print("Downloading flows...")
        flows = openml_call_with_retry(
            openml.flows.list_flows, output_format="dataframe"
        )
        if filter_fullname_regex:
            flows = flows[flows["full_name"].str.contains(filter_fullname_regex)]

        FLOWS_PATH.parents[0].mkdir(parents=True, exist_ok=True)
        flows.to_csv(FLOWS_PATH, index=False)
    else:
        print("Reading flows from cache...")
        flows = pd.read_csv(FLOWS_PATH)

    return flows


def download_tasks(
    num_tasks_to_process=None, offset=0, dataset_names=None, dataset_task_types=None
):
    print("Downloading tasks...")
    if dataset_names is None:
        return openml_call_with_retry(
            openml.tasks.list_tasks,
            output_format="dataframe",
            size=num_tasks_to_process,
            offset=offset,
        )

    tasks_dfs = []
    for i, dataset_name in enumerate(dataset_names):
        tasks_dfs.append(
            openml_call_with_retry(
                openml.tasks.list_tasks,
                output_format="dataframe",
                size=num_tasks_to_process,
                offset=offset,
                data_name=dataset_name,
                task_type=(
                    dataset_task_types[i] if dataset_task_types is not None else None
                ),
            )
        )

    tasks = pd.concat(tasks_dfs)
    fetched_dataset_names = tasks["name"].tolist()
    if set(fetched_dataset_names) != set(dataset_names):
        print(
            f"Warning: Tasks for datasets {set(dataset_names) - set(fetched_dataset_names)} not found"
        )

    return tasks


def get_tasks(
    num_tasks_to_process=10, offset=0, dataset_names=None, dataset_task_types=None
):
    if dataset_names is not None:
        print(
            f"Fetching {num_tasks_to_process} tasks for each of {len(dataset_names)} datasets: {dataset_names}..."
        )
    else:
        print(f"Fetching {num_tasks_to_process} tasks...")

    if not TASKS_PATH.exists():
        tasks = download_tasks(
            num_tasks_to_process, offset, dataset_names, dataset_task_types
        )

        TASKS_PATH.parents[0].mkdir(parents=True, exist_ok=True)
        tasks.to_csv(TASKS_PATH, index=False)
    else:
        tasks = pd.read_csv(TASKS_PATH)
        if (
            dataset_names is None and len(tasks) == num_tasks_to_process
        ):  # check if the tasks are already fetched
            print("Reading tasks from cache...")
            return tasks

        tasks = download_tasks(
            num_tasks_to_process, offset, dataset_names, dataset_task_types
        )

        if dataset_names is None:
            tasks.to_csv(TASKS_PATH, index=False)

    return tasks


def get_datasets(dataset_ids):
    print(f"Fetching {len(dataset_ids)} datasets...")
    if not DATASETS_PATH.exists():
        print("Downloading datasets...")
        datasets = openml_call_with_retry(
            openml.datasets.list_datasets,
            output_format="dataframe",
            data_id=dataset_ids,
        )
        DATASETS_PATH.parents[0].mkdir(parents=True, exist_ok=True)
        datasets.to_csv(DATASETS_PATH, index=False)
    else:
        print("Reading datasets from cache...")
        datasets = pd.read_csv(DATASETS_PATH)
        if len(datasets) != len(dataset_ids):
            print("Downloading datasets...")
            datasets = openml_call_with_retry(
                openml.datasets.list_datasets,
                output_format="dataframe",
                data_id=dataset_ids,
            )
            datasets.to_csv(DATASETS_PATH, index=False)

    return datasets


def create_data_entities(dataset, openml_task_type):
    print(f"Creating data entities for dataset {dataset.name}...")

    top_level_schema_namespace = KG_SCHEMAS["Data Science"]["namespace"]
    feature_data_entities = []
    label_data_entity = None
    ml_task_type = None
    for (
        id,
        feature,
    ) in dataset.features.items():
        feature_name_for_kg = feature.name
        if feature.name == dataset.default_target_attribute:  # label column
            if feature.data_type == "nominal":
                if openml_task_type == "Clustering":
                    ml_task_type = "Clustering"
                else:
                    # set ml_task_type only in case of classification because some ExeKG methods are compatible with multiple Classification subclasses
                    if len(feature.nominal_values) == 2:
                        ml_task_type = "BinaryClassification"
                    elif len(feature.nominal_values) > 2:
                        ml_task_type = "MulticlassClassification"
                    else:
                        raise ValueError(
                            f"Label column {feature.name} has less than 2 classes"
                        )

            label_data_entity = DataEntity(
                iri=top_level_schema_namespace
                + feature_name_for_kg
                + f"_dataset_{int(dataset.id)}",
                parent_entity=Entity(top_level_schema_namespace + "DataEntity"),
                source_value=feature_name_for_kg,
                data_semantics_iri=top_level_schema_namespace
                + OPENML_DATATYPE_TO_EXEKG_SEMANTICS[feature.data_type],
                data_structure_iri=top_level_schema_namespace + "Vector",
            )
            continue

        feature_data_entities.append(
            DataEntity(
                iri=top_level_schema_namespace
                + feature_name_for_kg
                + f"_dataset_{int(dataset.id)}",
                parent_entity=Entity(top_level_schema_namespace + "DataEntity"),
                source_value=feature_name_for_kg,
                data_semantics_iri=top_level_schema_namespace
                + OPENML_DATATYPE_TO_EXEKG_SEMANTICS[feature.data_type],
                data_structure_iri=top_level_schema_namespace + "Vector",
            )
        )

    if label_data_entity is None:
        raise ValueError("Label column not found in dataset")
    if len(feature_data_entities) == 0:
        raise ValueError("No feature columns found in dataset")

    return feature_data_entities, label_data_entity, ml_task_type


def openml_call_with_retry(method, *args, **kwargs):
    delay = INITIAL_REQ_DELAY
    while True:
        try:
            obj = method(*args, **kwargs)
            return obj
        except (
            requests.exceptions.ProxyError,
            openml.exceptions.OpenMLServerException,
        ):
            print(f"ProxyError encountered. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay = min(delay * 2, MAX_REQ_DELAY)  # exponential backoff with a cap


def get_task_runs_and_info(task_id, filter_flow_ids=None, num_runs_per_flow=None):
    print(f"Fetching info about task {task_id}...")
    task = openml_call_with_retry(
        openml.tasks.get_task, task_id, download_splits=False, **OPENML_DATASET_KWARGS
    )

    try:
        metric_name = METRIC_FOR_SORTING_PIPELINES_DICT[task.task_type]
    except KeyError:
        raise ValueError(f"Task type {task.task_type} not supported")

    run_ids, metric_values = get_best_runs_for_task(
        task_id,
        metric=metric_name,
        filter_flow_ids=filter_flow_ids,
        num_runs_per_flow=num_runs_per_flow,
    )

    dataset = openml_call_with_retry(
        openml.datasets.get_dataset, task.dataset_id, **OPENML_DATASET_KWARGS
    )

    # create data entities to add later to the ExeKG
    feature_data_entities, label_data_entity, ml_task_type = create_data_entities(
        dataset, task.task_type
    )

    est_procedure_sklearn, est_procedure_params_dict = get_est_procedure_from_task(task)

    run_ids_metric_values = (
        list(zip(run_ids, metric_values)) if run_ids and metric_values else []
    )

    print(
        f"""
    ---------------------------------
    Fetched info about task {task_id}
    ---------------------------------
    Task ID: {task_id}
    Dataset ID: {task.dataset_id}
    Task Type: {task.task_type}
    Metric: {metric_name}
    Number of Runs to Process: {len(run_ids) if run_ids else 0}
    Feature Data Entities: {', '.join([entity.name for entity in feature_data_entities])}
    Label Data Entity: {label_data_entity.name}
    Estimation Procedure: {est_procedure_sklearn}
    Estimation Procedure Parameters: {est_procedure_params_dict}
    ML Task Type: {ml_task_type}
    """
    )

    return (
        run_ids_metric_values,
        metric_name,
        dataset,
        feature_data_entities,
        label_data_entity,
        est_procedure_sklearn,
        est_procedure_params_dict,
        ml_task_type,
    )
