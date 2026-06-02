# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0


import pathlib
import re
import numpy as np
import torch
import pandas as pd
from typing import Any, List

from kge_experiments.utils.graph_utils import (
    get_pipeline_info_for_task,
)
from kge_experiments.utils.openml_utils import num_unique_dataset_ids


def get_pipeline_emb_rdf2vec(
    dataset_id,
    dataset_to_task_ids,
    exekgs_path,
    rdf2vec_kg,
    rdf2vec_model,
    exclude_flows_per_task,
):
    embeddings = []
    for task_id in dataset_to_task_ids[dataset_id]:
        pipeline_names, _, _ = get_pipeline_info_for_task(
            int(task_id), exekgs_path, exclude_flows_per_task
        )

        for pipeline_name in pipeline_names:
            pipeline_comp_names = [
                entity
                for entity in rdf2vec_kg.entities
                if entity.endswith(f"_{pipeline_name}")
                and "Concatenation" not in entity
                and "Method" in entity
            ]

            pipeline_comp_embeddings = []
            for pipeline_comp_name in pipeline_comp_names:
                pipeline_comp_embeddings.append(
                    rdf2vec_model.wv.get_vector(pipeline_comp_name)
                )

            avg_pipeline_comp_embedding = np.mean(pipeline_comp_embeddings, axis=0)
            embeddings.append(avg_pipeline_comp_embedding)

    return embeddings


def get_data_entity_emb_rdf2vec(
    entities, rdf2vec_kg, rdf2vec_model, num_failed_datasets
):
    entities_with_exekg_ns = [
        f"https://raw.githubusercontent.com/nsai-uio/ExeKGOntology/main/ds_exeKGOntology.ttl#{entity}"
        for entity in entities
    ]

    embeddings = []
    for entity in entities_with_exekg_ns:
        embeddings.append(rdf2vec_model.wv.get_vector(entity))

    if rdf2vec_kg.use_mlseakg:
        entities_with_mlseakg_ns = []
        for entity in entities:
            try:
                entities_with_mlseakg_ns.append(
                    f"http://w3id.org/mlsea/openml/feature/{rdf2vec_kg.exekg_de_id_to_mlseakg_feature_id[entity]}"
                )
            except KeyError as e:
                print(f"Data entity {entity} from ExeKGs not found in MLSeaKG")

        if not entities_with_mlseakg_ns:
            num_failed_datasets += 1

        for entity in entities_with_mlseakg_ns:
            embeddings.append(rdf2vec_model.wv.get_vector(entity))

    return embeddings, num_failed_datasets


def get_pipeline_emb_pykeen_lp(
    dataset_id,
    dataset_to_task_ids,
    exekgs_path,
    pykeen_lp_model,
    pykeen_lp_dataset,
    exclude_flows_per_task,
):
    """
    Get pipeline embeddings from a trained PyKEEN link prediction model.
    
    Args:
        dataset_id: Dataset ID
        dataset_to_task_ids: Mapping from dataset IDs to task IDs
        exekgs_path: Path to ExeKGs data
        pykeen_lp_model: Trained PyKEEN model
        pykeen_lp_dataset: PyKEEN dataset with entity_to_id mapping
        exclude_flows_per_task: Flows to exclude per task
        
    Returns:
        List of pipeline embeddings
    """
    entity_representation = pykeen_lp_model.entity_representations[0]
    entity_to_id = pykeen_lp_dataset.training.entity_to_id
    
    embeddings = []
    
    for task_id in dataset_to_task_ids[dataset_id]:
        pipeline_comp_ids = []
        pipeline_comp_embeddings = []
        pipeline_names, _, _ = get_pipeline_info_for_task(
            int(task_id), exekgs_path, exclude_flows_per_task
        )
        
        for pipeline_name in pipeline_names:
            # Find all method components of the pipeline
            for entity_name in entity_to_id.keys():
                entity_name_s = str(entity_name)
                if (
                    entity_name_s.endswith(f"_{pipeline_name}")
                    and "Concatenation" not in entity_name_s
                    and "Method" in entity_name_s
                ):
                    # is task (i.e. step) or method (i.e. algorithm) of pipeline
                    pipeline_comp_ids.append(entity_to_id[entity_name])
            
            for pipeline_comp_id in pipeline_comp_ids:
                pipeline_comp_embeddings.append(
                    entity_representation(torch.as_tensor(pipeline_comp_id).view(1))
                    .detach()
                    .cpu()
                    .numpy()
                    .flatten()
                    .ravel()
                )
            
            if pipeline_comp_embeddings:
                avg_pipeline_comp_embedding = np.mean(pipeline_comp_embeddings, axis=0)
                embeddings.append(avg_pipeline_comp_embedding)
    
    return embeddings


def get_data_entity_emb_pykeen_lp(
    entities, pykeen_lp_dataset, pykeen_lp_model, num_failed_datasets
):
    """
    Get data entity embeddings from a trained PyKEEN link prediction model.
    
    Args:
        entities: List of entity names
        pykeen_lp_dataset: PyKEEN dataset with entity_to_id mapping
        pykeen_lp_model: Trained PyKEEN model
        num_failed_datasets: Counter for failed datasets
        
    Returns:
        Tuple of (embeddings list, updated num_failed_datasets)
    """
    entity_representation = pykeen_lp_model.entity_representations[0]
    entity_to_id = pykeen_lp_dataset.training.entity_to_id
    
    # Try with ExeKG namespace
    entities_with_ns = [
        f"https://raw.githubusercontent.com/nsai-uio/ExeKGOntology/main/ds_exeKGOntology.ttl#{entity}"
        for entity in entities
    ]
    
    embeddings = []
    try:
        entity_ids = [entity_to_id[entity] for entity in entities_with_ns]
    except KeyError as e:
        print(f"KeyError: {e} - Entity not found in PyKEEN LP model")
        num_failed_datasets += 1
        return embeddings, num_failed_datasets
    
    for entity_id in entity_ids:
        embeddings.append(
            entity_representation(torch.as_tensor(entity_id).view(1))
            .detach()
            .cpu()
            .numpy()
            .flatten()
            .ravel()
        )
    
    return embeddings, num_failed_datasets


def get_dataset_info(
    openml_datasets_df: pd.DataFrame,
    exekgs_path: pathlib.Path,
    exclude_flows_per_task: dict = None,
):
    columns = {}
    nums_of_columns = {}
    flows = {}
    nums_of_flows = {}
    for index, row in openml_datasets_df.iterrows():
        task_id = row["task_id"]
        dataset_id = row["dataset_id"]
        feature_entities = row["feature_data_entities"].split(", ")
        label_entity = row["label_data_entity"]
        data_entities = feature_entities + [label_entity]

        columns[dataset_id] = data_entities
        nums_of_columns[dataset_id] = len(data_entities)

        _, flow_ids, _ = get_pipeline_info_for_task(
            int(task_id), exekgs_path, exclude_flows_per_task
        )
        if not flow_ids:
            continue

        if dataset_id not in flows:
            flows[dataset_id] = set()
        flows[dataset_id].update(flow_ids)

        if dataset_id not in nums_of_flows:
            nums_of_flows[dataset_id] = 0
        nums_of_flows[dataset_id] += 1

    return columns, nums_of_columns, flows, nums_of_flows


def get_dataset_emb(
    dataset_to_task_ids: dict,
    rdf2vec_kg: Any,
    rdf2vec_model: Any,
    openml_datasets_df: pd.DataFrame,
    exekgs_path: pathlib.Path,
    kge_type: str,
    stored_pipeline_emb_rdf2vec_dict: dict = {},
    stored_data_entity_emb_rdf2vec_dict: dict = {},
    use_pipeline_embeddings: bool = False,
    use_data_entity_embeddings: bool = True,
    exclude_flows_per_task: dict = None,
) -> dict:
    """Calculate the average embeddings for each dataset."""
    print(
        f"Calculating embeddings for {num_unique_dataset_ids(openml_datasets_df)} datasets. use_pipeline_embeddings: {use_pipeline_embeddings}, use_data_entity_embeddings: {use_data_entity_embeddings}"
    )
    avg_embeddings = {}
    num_datasets_not_found_in_mlseakg = 0
    for index, row in openml_datasets_df.iterrows():
        dataset_id = row["dataset_id"]
        feature_entities = row["feature_data_entities"].split(", ")
        label_entity = row["label_data_entity"]
        data_entities = feature_entities + [label_entity]

        embeddings = []
        if use_pipeline_embeddings:
            if dataset_id in stored_pipeline_emb_rdf2vec_dict:
                pipeline_embeddings = stored_pipeline_emb_rdf2vec_dict[dataset_id]
            else:
                pipeline_embeddings = get_pipeline_emb_rdf2vec(
                    dataset_id,
                    dataset_to_task_ids,
                    exekgs_path,
                    rdf2vec_kg,
                    rdf2vec_model,
                    exclude_flows_per_task,
                )
                stored_pipeline_emb_rdf2vec_dict[dataset_id] = pipeline_embeddings

            avg_pipelines_embedding = (
                np.mean(pipeline_embeddings, axis=0)
                if len(pipeline_embeddings) > 0
                else None
            )
            embeddings.append(avg_pipelines_embedding)

        if use_data_entity_embeddings:
            data_entity_embeddings, new_num_failed_datasets = (
                get_data_entity_emb_rdf2vec(
                    data_entities,
                    rdf2vec_kg,
                    rdf2vec_model,
                    num_datasets_not_found_in_mlseakg,
                )
            )

            if new_num_failed_datasets > num_datasets_not_found_in_mlseakg:
                print(f"Failed dataset: {dataset_id}")
                num_datasets_not_found_in_mlseakg = new_num_failed_datasets
                continue

            avg_data_entities_embedding = (
                np.mean(data_entity_embeddings, axis=0)
                if len(data_entity_embeddings) > 0
                else None
            )
            embeddings.append(avg_data_entities_embedding)

        if len(embeddings) == 0:
            print(f"No embeddings found for {dataset_id}")
            continue

        if all([embedding is not None for embedding in embeddings]):
            avg_embedding = np.mean(embeddings, axis=0)
        else:
            avg_embedding = None
        avg_embeddings[dataset_id] = avg_embedding

    if use_data_entity_embeddings:
        print(
            f"Number of failed datasets: {num_datasets_not_found_in_mlseakg} out of {num_unique_dataset_ids(openml_datasets_df)}"
        )

    return avg_embeddings
