# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0


from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import pathlib
import pickle
import sys
import time
import networkx as nx
import numpy as np
import multiprocessing

from tqdm import tqdm
from kge_experiments.config import OUTPUT_DIR
from kge_experiments.utils.graph_utils import (
    load_graphs_for_task,
)


def calculate_ged(graph1, graph2, upper_bound=None):
    """Calculate graph edit distance with a timeout using multiprocessing."""
    ged = next(nx.optimize_graph_edit_distance(graph1, graph2, upper_bound=upper_bound))

    return ged


def calculate_avg_ged_between_exekgs(
    task_id1,
    task_id2,
    task_to_runs_to_loaded_graphs,
    exclude_flows_per_task,
    lock,
):
    def load_graphs_if_needed(task_id):
        with lock:
            if task_id in task_to_runs_to_loaded_graphs:
                return task_to_runs_to_loaded_graphs[task_id]
        loaded_graphs = load_graphs_for_task(task_id, exclude_flows_per_task)
        if loaded_graphs is not None:
            with lock:
                task_to_runs_to_loaded_graphs[task_id] = loaded_graphs
        return loaded_graphs

    run_to_loaded_graph1 = load_graphs_if_needed(task_id1)
    if run_to_loaded_graph1 is None:
        return None

    run_to_loaded_graph2 = load_graphs_if_needed(task_id2)
    if run_to_loaded_graph2 is None:
        return None

    ged_values = []
    for run_id1, graph1 in run_to_loaded_graph1.items():
        for run_id2, graph2 in run_to_loaded_graph2.items():
            ged = calculate_ged(graph1, graph2)
            ged_values.append(ged)

    if ged_values:
        avg_ged = np.mean(ged_values)
    else:
        avg_ged = None

    return avg_ged


def calculate_graph_edit_distances(
    dataset_pairs: list,
    shared_graphs: dict,
    lock,
    dataset_to_task_ids: dict,
    exclude_flows_per_task: dict,
) -> list:
    """Calculate graph edit distances between graphs of OpenML tasks."""
    graph_edit_distances = []

    for dataset_id1, dataset_id2 in dataset_pairs:
        task_ids1 = dataset_to_task_ids[dataset_id1]
        task_ids2 = dataset_to_task_ids[dataset_id2]

        ged_values = []
        for task_id1 in task_ids1:
            for task_id2 in task_ids2:
                if task_id1 != task_id2:
                    ged = calculate_avg_ged_between_exekgs(
                        int(task_id1),
                        int(task_id2),
                        shared_graphs,
                        exclude_flows_per_task,
                        lock,
                    )
                    if ged is not None:
                        ged_values.append(ged)

        if ged_values:
            avg_ged = np.mean(ged_values)
        else:
            avg_ged = None

        graph_edit_distances.append((dataset_id1, dataset_id2, avg_ged))

    return graph_edit_distances


def remove_already_processed_pairs(dataset_combinations, similarities_df):
    print("Filtering dataset combinations based on similarities_df ...")
    similarities_no_ged_df = similarities_df[
        similarities_df["Avg Pairwise GED"].isnull()
    ]

    if len(similarities_no_ged_df) == 0:
        return []

    remaining_combinations = similarities_no_ged_df[
        ["Dataset 1", "Dataset 2"]
    ].values.tolist()

    print(
        f"Found {len(similarities_df) - len(remaining_combinations)} already calculated GEDs. Filtering ..."
    )

    dataset_combinations = [
        dataset_pair
        for dataset_pair in dataset_combinations
        if dataset_pair in remaining_combinations
    ]

    print(
        f"Dataset pairs remained after filtering step 1: {len(dataset_combinations)} ..."
    )

    if (OUTPUT_DIR / "geds.pkl").exists():
        print("Filtering dataset combinations based on geds.pkl ...")
        geds = pickle.load(open(OUTPUT_DIR / "geds.pkl", "rb"))
        existing_combinations = set(
            (ged[0], ged[1]) for ged_list in geds for ged in ged_list
        )
        print(
            f"Found {len(existing_combinations)} already calculated GEDs. Filtering ..."
        )
        dataset_combinations = [
            dataset_pair
            for dataset_pair in dataset_combinations
            if (dataset_pair[0], dataset_pair[1]) not in existing_combinations
        ]
        print(
            f"Dataset pairs remained after filtering step 2: {len(dataset_combinations)} ..."
        )

    return dataset_combinations


def calculate_geds_for_dataset_combinations(
    dataset_combinations,
    similarities_df,
    dataset_to_task_ids,
    exclude_flows_per_task,
    num_processes,
    chunk_size,
):
    dataset_combinations = remove_already_processed_pairs(
        dataset_combinations, similarities_df
    )
    if len(dataset_combinations) == 0:
        print("All GEDs have already been calculated.")
        return []

    print(f"Chunking dataset combinations using chunk size: {chunk_size} ...")
    chunks = [
        dataset_combinations[i : i + chunk_size]
        for i in range(0, len(dataset_combinations), chunk_size)
    ]

    results = []

    with multiprocessing.Manager() as manager:
        shared_graphs = manager.dict()
        lock = manager.Lock()
        with tqdm(
            total=len(dataset_combinations), desc="Calculating GEDs", unit="chunk"
        ) as pbar:
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = [
                    executor.submit(
                        calculate_graph_edit_distances,
                        chunk,
                        shared_graphs,
                        lock,
                        dataset_to_task_ids,
                        exclude_flows_per_task,
                    )
                    for chunk in chunks
                ]
                for future in as_completed(futures):
                    pbar.update(chunk_size)
                    results.append(future.result())
                    pickle.dump(results, open(OUTPUT_DIR / "geds.pkl", "wb"))

    graph_edit_distances = [item for sublist in results for item in sublist]
    return graph_edit_distances
