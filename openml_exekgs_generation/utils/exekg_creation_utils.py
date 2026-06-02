# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from collections import OrderedDict
from copy import copy, deepcopy
import time

import openml
from rdflib import URIRef
from openml_exekgs_generation.config import (
    EXEKGS_OUTPUT_DIR,
    LOG_DIR,
    METRIC_FOR_SORTING_PIPELINES_DICT,
    OPENML_METRIC_NAME_TO_EXEKG,
    PERFORMANCE_CALCULATION_KWARGS_DICT,
    RUNS_LOG_FILENAME,
)

from exe_kg_lib.utils.kg_validation_utils import KGValidationError

from openml_exekgs_generation.utils.exec_utils import (
    init_runs_log_dict,
    log_new_runs,
    update_runs_log_dict,
)
from openml_exekgs_generation.utils.openml_utils import openml_call_with_retry
from openml_exekgs_generation.utils.string_utils import get_param_dict_from_openml_run
from openml_exekgs_generation.utils.query_utils import get_lower_and_upper_task_of_method
from openml_exekgs_generation.utils.string_utils import (
    code_method_to_exekg_method,
    get_method_name_and_params_dict,
)


def add_concatenation_task(exe_kg, feature_data_entities):
    concatenate_task = exe_kg.add_task(
        kg_schema_short="ml",
        input_entity_dict={"DataInConcatenation": feature_data_entities},
        method_type="ConcatenationMethod",
        method_params_dict={},
        task_type="Concatenation",
    )

    return concatenate_task.output_dict["DataOutConcatenatedData"]


def add_datasplitting_task(
    exe_kg,
    concatenated_data_entity,
    label_data_entity,
    est_procedure_sklearn,
    est_procedure_params_dict,
):
    data_splitting_task = exe_kg.add_task(
        kg_schema_short="ml",
        task_type="DataSplitting",
        input_entity_dict={
            "DataInDataSplittingX": [concatenated_data_entity],
            "DataInDataSplittingY": [label_data_entity],
        },
        method_type=est_procedure_sklearn,
        method_params_dict=est_procedure_params_dict,
    )

    return (
        data_splitting_task.output_dict["DataOutSplittedTrainDataX"],
        data_splitting_task.output_dict["DataOutSplittedTrainDataY"],
        data_splitting_task.output_dict["DataOutSplittedTestDataX"],
        data_splitting_task.output_dict["DataOutSplittedTestDataY"],
    )


def add_train_and_test_tasks(
    exe_kg,
    train_task_type,
    train_x,
    train_real_y,
    test_x,
    model_name,
    model_params_dict,
    submodel_name=None,
    submodel_params_dict=None,
):
    input_entity_dict = {
        "DataInTrainX": [train_x],
        "DataInTrainY": [train_real_y],
    }

    if submodel_name:
        input_entity_dict["InputModelAsMethod"] = exe_kg.create_method(
            method_type=submodel_name,
            params_dict=submodel_params_dict if submodel_params_dict else {},
        )

    # train task
    train_task = exe_kg.add_task(
        kg_schema_short="ml",
        task_type=train_task_type,
        input_entity_dict=input_entity_dict,
        method_type=model_name,
        method_params_dict=model_params_dict,
    )
    model = train_task.output_dict["DataOutTrainModel"]

    # test task
    test_task = exe_kg.add_task(
        kg_schema_short="ml",
        task_type="Test",
        input_entity_dict={
            "DataInTestModel": [model],
            "DataInTestX": [test_x],
        },
        method_type="TestMethod",
        method_params_dict={},
    )
    test_predicted_y = test_task.output_dict["DataOutPredictedValueTest"]

    return test_predicted_y


def add_prepare_transformer_and_transform_tasks(
    exe_kg,
    prepare_transformer_task_type,
    data_to_transform,
    model_name,
    model_params_dict,
):
    # prepare transformer task
    prepare_transformer_task = exe_kg.add_task(
        kg_schema_short="ml",
        task_type=prepare_transformer_task_type,
        input_entity_dict={
            "DataInToPrepareTransformer": [data_to_transform],
        },
        method_type=model_name,
        method_params_dict=model_params_dict,
    )
    transformer = prepare_transformer_task.output_dict["DataOutTransformer"]

    # transform task
    transform_task = exe_kg.add_task(
        kg_schema_short="ml",
        task_type="Transform",
        input_entity_dict={
            "DataInToTransform": [data_to_transform],
            "DataInTransformer": [transformer],
        },
        method_type="TransformMethod",
        method_params_dict={},
    )
    data_transformed = transform_task.output_dict["DataOutTransformed"]

    return data_transformed


def add_performance_calculation_task(
    exe_kg,
    real_y,
    predicted_y,
    metric_method_name,
    metric_params_dict,
):
    performance_calc_task = exe_kg.add_task(
        kg_schema_short="ml",
        task_type="PerformanceCalculation",
        input_entity_dict={
            "DataInRealY": [real_y],
            "DataInPredictedY": [predicted_y],
        },
        method_type=metric_method_name,
        method_params_dict=metric_params_dict,
    )
    return performance_calc_task.output_dict["DataOutScore"]


def create_dummy_features_and_label_data_entities(exe_kg):
    feature_columns = [
        "feature_1",
        "feature_2",
        "feature_3",
    ]
    label_column = "label"

    feature_data_entities = []
    for feature_column in feature_columns:
        feature_data_entities.append(
            exe_kg.create_data_entity(
                name=feature_column,
                source_value=feature_column,
                data_semantics_name="Numerical",
                data_structure_name="Vector",
            )
        )

    label_data_entity = exe_kg.create_data_entity(
        name=label_column,
        source_value=label_column,
        data_semantics_name="Categorical",
        data_structure_name="Vector",
    )

    return feature_data_entities, label_data_entity


def add_or_update_task(
    exe_kg,
    pipeline_name,
    run,
    component_obj,
    ml_task_type,
    train_x,
    test_x,
    train_real_y,
    train_x_transformed_last=None,
    test_x_transformed_last=None,
    subcomponent_obj=None,
    update_mode="ADD_TASKS",
):
    method_name, method_params_dict = get_method_name_and_params_dict(
        run, component_obj
    )
    submethod_name = None
    submethod_params_dict = None
    if subcomponent_obj:
        submethod_name, submethod_params_dict = get_method_name_and_params_dict(
            run, subcomponent_obj
        )

    method_lower_task_name, method_upper_task_name = get_lower_and_upper_task_of_method(
        exe_kg, method_name
    )
    if method_lower_task_name is None or method_upper_task_name is None:
        raise ValueError(f"Method {method_name} not found in the KG schemata.")

    ml_task_type = (
        ml_task_type if ml_task_type else method_lower_task_name
    )  # else: the first task that was found to be connected with the current method

    if (
        method_name == "IsolationForestMethod"
        and "classification" in ml_task_type.lower()
    ):
        raise ValueError(
            f"Method {method_name} is not supported for classification tasks."
        )

    test_predicted_y = None
    if method_upper_task_name == "Train":
        if update_mode == "ADD_TASKS":
            test_predicted_y = add_train_and_test_tasks(
                exe_kg,
                ml_task_type,
                (
                    train_x
                    if train_x_transformed_last is None
                    else train_x_transformed_last
                ),
                train_real_y,
                (
                    test_x
                    if test_x_transformed_last is None
                    else test_x_transformed_last
                ),
                method_name,
                method_params_dict,
                submethod_name,
                submethod_params_dict,
            )
        elif update_mode == "UPDATE_TASK_PARAMS" and run:
            exe_kg.update_param_values(
                {
                    (
                        "ml",
                        f"{method_name}1_{pipeline_name}",  # NOTE: "1" is static as there is only one train task (and method) in each pipeline
                    ): method_params_dict
                }
            )
            if submethod_params_dict:
                exe_kg.update_param_values(
                    {
                        (
                            "ml",
                            f"{submethod_name}1_{pipeline_name}",  # NOTE: "1" is static as there is only one train task (and method) in each pipeline
                        ): submethod_params_dict,
                    }
                )
    elif method_upper_task_name == "PrepareTransformer":
        if update_mode == "ADD_TASKS":
            train_x_transformed_last = add_prepare_transformer_and_transform_tasks(
                exe_kg,
                method_lower_task_name,
                (
                    train_x
                    if train_x_transformed_last is None
                    else train_x_transformed_last
                ),
                method_name,
                method_params_dict,
            )
            test_x_transformed_last = add_prepare_transformer_and_transform_tasks(
                exe_kg,
                method_lower_task_name,
                test_x if test_x_transformed_last is None else test_x_transformed_last,
                method_name,
                method_params_dict,
            )
        elif update_mode == "UPDATE_TASK_PARAMS" and run:
            # NOTE: each method is assumed to be used twice in the pipeline (for train and test data)
            exe_kg.update_param_values(
                {
                    (
                        "ml",
                        f"{method_name}1_{pipeline_name}",
                    ): method_params_dict,
                    (
                        "ml",
                        f"{method_name}2_{pipeline_name}",
                    ): method_params_dict,
                }
            )
    else:
        raise ValueError(
            f"Upper task found for method {method_name} is {method_upper_task_name} and cannot be handled."
        )

    return (
        test_predicted_y,
        train_x_transformed_last,
        test_x_transformed_last,
        method_lower_task_name,
    )


def add_or_update_tasks(
    exe_kg,
    pipeline_name,
    components,
    flow,
    run,
    ml_task_type,
    train_x,
    test_x,
    train_real_y,
    train_x_transformed_last=None,
    test_x_transformed_last=None,
    depth=1,
    update_mode="ADD_TASKS",
):
    components_ordered = OrderedDict(
        sorted(components.items(), key=lambda x: flow.name.index(x[1].class_name))
    )
    components_ordered_items = list(components_ordered.items())

    # add tasks to the KG based on the flow's details
    for i, (component_name, component_obj) in enumerate(components_ordered_items):
        subcomponents = component_obj.components
        if not subcomponents:  # or estimator_subcomponent is not None:
            (
                test_predicted_y,
                train_x_transformed_last,
                test_x_transformed_last,
                ml_method_lower_task_name,
            ) = add_or_update_task(
                exe_kg,
                pipeline_name,
                run,
                component_obj,
                ml_task_type,
                train_x,
                test_x,
                train_real_y,
                train_x_transformed_last,
                test_x_transformed_last,
                None,
                update_mode,
            )
            continue

        (
            test_predicted_y,
            train_x_transformed_last,
            test_x_transformed_last,
            ml_method_lower_task_name,
        ) = add_ensemble_or_model_selection_task(
            component_obj,
            subcomponents,
            ml_task_type,
            exe_kg,
            pipeline_name,
            run,
            train_x,
            test_x,
            train_real_y,
            update_mode,
        )

        if test_predicted_y:
            continue

        (
            test_predicted_y,
            train_x_transformed_last,
            test_x_transformed_last,
            ml_method_lower_task_name,
        ) = add_or_update_tasks(
            exe_kg,
            pipeline_name,
            subcomponents,
            component_obj,
            run,
            ml_task_type,
            train_x,
            test_x,
            train_real_y,
            train_x_transformed_last,
            test_x_transformed_last,
            depth + 1,
            update_mode,
        )

    return (
        test_predicted_y,
        train_x_transformed_last,
        test_x_transformed_last,
        ml_method_lower_task_name,
    )


def add_ensemble_or_model_selection_task(
    flow,
    subcomponents,
    ml_task_type,
    exe_kg,
    pipeline_name,
    run,
    train_x,
    test_x,
    train_real_y,
    update_mode="ADD_TASKS",
):
    estimator_subcomponent = None
    if subcomponents:
        estimator_subcomponent = subcomponents.get("estimator", None)
        if estimator_subcomponent is None:
            estimator_subcomponent = subcomponents.get("base_estimator", None)

    if (
        estimator_subcomponent is None
        or "sklearn.pipeline.Pipeline" in estimator_subcomponent.class_name + flow.name
    ):
        return None, None, None, None

    if "model_selection" in flow.name and estimator_subcomponent:
        (
            test_predicted_y,
            train_x_transformed_last,
            test_x_transformed_last,
            ml_method_lower_task_name,
        ) = add_or_update_task(
            exe_kg,
            pipeline_name,
            run,
            flow,
            "ModelSelection",
            train_x,
            test_x,
            train_real_y,
            None,
            None,
            estimator_subcomponent,
            update_mode,
        )
    elif "ensemble" in flow.name and estimator_subcomponent:
        (
            test_predicted_y,
            train_x_transformed_last,
            test_x_transformed_last,
            ml_method_lower_task_name,
        ) = add_or_update_task(
            exe_kg,
            pipeline_name,
            run,
            flow,
            ml_task_type,
            train_x,
            test_x,
            train_real_y,
            None,
            None,
            estimator_subcomponent,
            update_mode,
        )
    else:
        raise ValueError(
            f"Flow {flow.id} uses {flow.class_name} and is not a model selection or ensemble method but has a subcomponent {estimator_subcomponent.class_name}."
        )

    return (
        test_predicted_y,
        train_x_transformed_last,
        test_x_transformed_last,
        ml_method_lower_task_name,
    )


def init_exekg(
    exe_kg,
    dataset,
    feature_data_entities=None,
    label_data_entity=None,
    est_procedure_sklearn=None,
    est_procedure_params_dict=None,
):
    print([e.name for e in feature_data_entities])
    if not feature_data_entities or not label_data_entity:
        feature_data_entities, label_data_entity = (
            create_dummy_features_and_label_data_entities(exe_kg)
        )

    for data_entity in feature_data_entities + [label_data_entity]:
        exe_kg.pipeline_serializable.add_data_entity(
            data_entity.name,
            data_entity.source,
            data_entity.data_semantics,
            data_entity.data_structure,
        )

    input_data_path = "path/to/example"
    if dataset and dataset.parquet_file:
        input_data_path = dataset.parquet_file
    elif dataset and dataset.name:
        input_data_path = f"path/to/{dataset.name}"

    exe_kg.create_pipeline_task(
        "ToBeReplaced",
        input_data_path=input_data_path,
        plots_output_dir="path/to/plots/output/dir",
    )

    concatenated_features_data_entity = add_concatenation_task(
        exe_kg, feature_data_entities
    )

    # add estimation procedure task to exe_kg
    train_x, train_real_y, test_x, test_real_y = add_datasplitting_task(
        exe_kg,
        concatenated_features_data_entity,
        label_data_entity,
        (
            code_method_to_exekg_method(est_procedure_sklearn)
            if est_procedure_sklearn
            else code_method_to_exekg_method("TrainTestSplit")
        ),
        est_procedure_params_dict if est_procedure_params_dict else {},
    )

    return train_x, train_real_y, test_x, test_real_y


def update_exekg(
    exe_kg,
    train_x,
    train_real_y,
    test_x,
    test_real_y,
    flow=None,
    run=None,
    task_id=None,
    ml_task_type=None,
    metric_value=None,
    update_mode="ADD_TASKS",  # or "UPDATE_TASK_PARAMS"
    save=True,
):
    pipeline_name = "pipeline"
    if task_id:
        pipeline_name += f"_task_{task_id}"
    if flow:
        pipeline_name += f"_flow_{flow.id}"
    if run:
        pipeline_name += f"_run_{run.id}"

    old_name, new_name = exe_kg.update_pipeline_name(pipeline_name)
    train_x.iri = URIRef(train_x.iri.replace(old_name, pipeline_name))
    train_real_y.iri = URIRef(train_real_y.iri.replace(old_name, pipeline_name))
    test_x.iri = URIRef(test_x.iri.replace(old_name, pipeline_name))
    test_real_y.iri = URIRef(test_real_y.iri.replace(old_name, pipeline_name))
    exe_kg.last_created_task.iri = URIRef(
        exe_kg.last_created_task.iri.replace("ToBeReplaced", pipeline_name)
    )

    if not flow.components:  # assume no preprocessing task in the flow
        method_name = code_method_to_exekg_method(flow.name.split(".")[-1])
        ml_method_lower_task_name, method_upper_task_name = (
            get_lower_and_upper_task_of_method(exe_kg, method_name)
        )
        ml_task_type = ml_task_type if ml_task_type else ml_method_lower_task_name # else: the first task that was found to be connected with the current method
        if update_mode == "ADD_TASKS":
            test_predicted_y = add_train_and_test_tasks(
                exe_kg,
                ml_task_type,  
                train_x,
                train_real_y,
                test_x,
                method_name,
                get_param_dict_from_openml_run(run) if run else {},
            )
        elif update_mode == "UPDATE_TASK_PARAMS" and run:
            exe_kg.update_param_values(
                {
                    (
                        "ml",
                        f"{method_name}1_{pipeline_name}",  # TODO: make "1" dynamic
                        # TODO: params are updated but old ones are not removed. e.g. if task has "n_estimators", "max_depth" and "n_estimators" is updated to 100, "max_depth" is still in the KG
                    ): get_param_dict_from_openml_run(run),
                }
            )
    else:
        (
            test_predicted_y,
            train_x_transformed_last,
            test_x_transformed_last,
            ml_method_lower_task_name,
        ) = add_ensemble_or_model_selection_task(
            flow,
            flow.components,
            ml_task_type,
            exe_kg,
            pipeline_name,
            run,
            train_x,
            test_x,
            train_real_y,
            update_mode,
        )

        if test_predicted_y is None:
            (
                test_predicted_y,
                _,
                _,
                ml_method_lower_task_name,
            ) = add_or_update_tasks(
                exe_kg,
                pipeline_name,
                flow.components,
                flow,
                run,
                ml_task_type,
                train_x,
                test_x,
                train_real_y,
                None,
                None,
                1,
                update_mode,
            )

    openml_metric_name = (
        METRIC_FOR_SORTING_PIPELINES_DICT["Supervised Classification"]
        if "Classification" in ml_method_lower_task_name
        else METRIC_FOR_SORTING_PIPELINES_DICT["Supervised Regression"]
    )
    metric_method_name = code_method_to_exekg_method(
        OPENML_METRIC_NAME_TO_EXEKG[openml_metric_name]
    )

    if update_mode == "ADD_TASKS":
        add_performance_calculation_task(
            exe_kg,
            test_real_y,
            test_predicted_y,
            metric_method_name,
            PERFORMANCE_CALCULATION_KWARGS_DICT[openml_metric_name],
        )
    elif update_mode == "UPDATE_TASK_PARAMS" and metric_value:
        exe_kg.update_metric_values(
            {
                f"DataOutScore_PerformanceCalculation1_{pipeline_name}_{metric_method_name}": float(
                    metric_value
                )
            }
        )

    if save:
        out_dir_path = EXEKGS_OUTPUT_DIR
        if task_id:
            out_dir_path = out_dir_path / f"task_{task_id}"
        if flow:
            out_dir_path = out_dir_path / f"flow_{flow.id}"
        if run:
            out_dir_path = out_dir_path / f"run_{run.id}"

        # Save the KG
        exe_kg.apply_changes_to_ttl(out_dir_path, check_executability=False)

    return pipeline_name


def create_exekgs_from_runs(
    run_ids_metric_values,
    base_exe_kg,
    train_x,
    train_real_y,
    test_x,
    test_real_y,
    task_id,
    ml_task_type,
    lock=None,
):
    runs_log_path = LOG_DIR / RUNS_LOG_FILENAME
    processed_run_infos_dict = init_runs_log_dict()

    base_graph = (
        base_exe_kg.exe_kg
    )
    base_last_created_task = base_exe_kg.last_created_task
    base_task_type_dict = base_exe_kg.task_type_dict
    base_method_type_dict = base_exe_kg.method_type_dict

    runs = [
        openml_call_with_retry(openml.runs.get_run, run_id)
        for run_id, _ in run_ids_metric_values
    ]
    flows = [openml_call_with_retry(openml.flows.get_flow, run.flow_id) for run in runs]
    metric_values = [metric_value for _, metric_value in run_ids_metric_values]

    runs_flows_metric_values = zip(runs, flows, metric_values)

    # group by flow.id
    runs_flows_metric_values_dict = {}
    for run, flow, metric_value in runs_flows_metric_values:
        if flow.id not in runs_flows_metric_values_dict:
            runs_flows_metric_values_dict[flow.id] = []
        runs_flows_metric_values_dict[flow.id].append((run, flow, metric_value))

    flows_processed = set()
    num_runs_processed = 0
    total_runs_processed = 0
    total_time_taken = 0.0
    for flow_id, run_info_l in runs_flows_metric_values_dict.items():
        for run, flow, metric_value in run_info_l:
            num_runs_processed += 1

            start = time.time()

            if flow.id not in flows_processed:
                base_exe_kg.exe_kg = deepcopy(base_graph)
                base_exe_kg.last_created_task = deepcopy(base_last_created_task)
                base_exe_kg.task_type_dict = copy(base_task_type_dict)
                base_exe_kg.method_type_dict = copy(base_method_type_dict)

                try:
                    update_exekg(
                        base_exe_kg,
                        train_x,
                        train_real_y,
                        test_x,
                        test_real_y,
                        flow,
                        None,
                        task_id,
                        ml_task_type,
                        metric_value,
                        update_mode="ADD_TASKS",
                        save=False,
                    )
                    flows_processed.add(flow.id)
                except ValueError as e:
                    print(
                        f"{e}\nError creating ExeKG for run {run.id} and flow {flow.id} with name {flow.name}..."
                    )
                    update_runs_log_dict(
                        processed_run_infos_dict,
                        task_id,
                        run.id,
                        flow.id,
                        flow.name,
                        None,
                        time.time() - start,
                        str(e),
                    )
                    break  # flow is erroneous, skip the rest of the runs for this flow

            print(
                f"\n--> Task: {task_id}, Flow: {flow.id}, Run: {run.id}, Flow name: {flow.name}\n"
            )

            try:
                pipeline_name = update_exekg(
                    base_exe_kg,
                    train_x,
                    train_real_y,
                    test_x,
                    test_real_y,
                    flow,
                    run,
                    task_id,
                    ml_task_type,
                    metric_value,
                    update_mode="UPDATE_TASK_PARAMS",
                )
                time_taken = time.time() - start
                total_runs_processed += 1
                total_time_taken += time_taken
                update_runs_log_dict(
                    processed_run_infos_dict,
                    task_id,
                    run.id,
                    flow.id,
                    flow.name,
                    pipeline_name,
                    time_taken,
                    None,
                )
                print(f"Processed run {run.id} in {time.time() - start:.2f} seconds.")
            except (ValueError, KGValidationError) as e:
                print(
                    f"{e}\nError creating ExeKG for run {run.id} and flow {flow.id} with name {flow.name}..."
                )
                update_runs_log_dict(
                    processed_run_infos_dict,
                    task_id,
                    run.id,
                    flow.id,
                    flow.name,
                    None,
                    time.time() - start,
                    str(e),
                )

        if (
            num_runs_processed % 10 == 0
            or num_runs_processed == len(run_ids_metric_values) - 1
        ):
            msg = f"Processed {num_runs_processed + 1} runs out of {len(run_ids_metric_values)}."
            if total_time_taken > 0:
                speed_runs_per_min = total_runs_processed / (total_time_taken / 60)
                msg += f" Speed: {speed_runs_per_min:.2f} runs/min."
            print(msg)

            log_new_runs(
                runs_log_path,
                processed_run_infos_dict,
                lock,
            )

            processed_run_infos_dict = init_runs_log_dict()
