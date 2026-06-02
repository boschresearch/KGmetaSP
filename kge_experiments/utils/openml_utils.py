# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0


from collections import defaultdict
import openml
import pandas as pd

from kge_experiments.config import DATASETS_CSV_PATH


def get_flow_name(flow_id):
    """Fetch the flow name from OpenML given a flow ID."""
    try:
        flow = openml.flows.get_flow(flow_id)
        return flow.name
    except Exception as e:
        print(f"Error fetching flow {flow_id} from OpenML: {e}")
        return None


def num_unique_dataset_ids(tasks_log_df):
    return len(tasks_log_df["dataset_id"].unique())


def read_task_info_from_logs(filter_by_dataset_ids=None):
    df = pd.read_csv(DATASETS_CSV_PATH)
    df = df[df["error"].isnull() & df["num_of_exekgs"] > 0]

    if filter_by_dataset_ids:
        print(f"Filtering datasets by {len(filter_by_dataset_ids)} predefined IDs ...")
        df = df[df["dataset_id"].isin(filter_by_dataset_ids)]

        if len(df) == 0:
            print("No datasets found with the predefined IDs")
            exit(1)

        if num_unique_dataset_ids(df) < len(filter_by_dataset_ids):
            print(
                f"{len(filter_by_dataset_ids) - num_unique_dataset_ids(df)} datasets were not found with the predefined IDs. The following datasets were not found:"
            )
            print(set(filter_by_dataset_ids) - set(df["dataset_id"].tolist()))

    return df


def create_dataset_to_task_ids(tasks_df):
    dataset_to_task_ids = defaultdict(list)
    for dataset_id, task_id in tasks_df[["dataset_id", "task_id"]].values:
        dataset_to_task_ids[dataset_id].append(task_id)
    return dict(dataset_to_task_ids)


def statistics_on_openml_tasks(tasks_df):
    print("Number of tasks:", len(tasks_df))
    print("Number of datasets:", len(tasks_df["dataset_id"].unique()))
    print("Number of runs:", tasks_df["num_of_exekgs"].sum())

    print("Task types:", tasks_df["exekg_train_task_type"].unique())
    print("Number of datasets per task type:")
    print(
        tasks_df.drop_duplicates(subset=["dataset_id"])[
            "metric_to_sort_runs"
        ].value_counts()
    )
    print(tasks_df[tasks_df["metric_to_sort_runs"] == "root_mean_squared_error"]["dataset_id"].unique())
    print(
        "Number of estimation procedures:",
        len(tasks_df["est_procedure"].unique()),
    )
    print("Estimation procedures:", tasks_df["est_procedure"].unique())
    print("Number of metrics:", len(tasks_df["metric_to_sort_runs"].unique()))
    print("Metrics:", tasks_df["metric_to_sort_runs"].unique())
