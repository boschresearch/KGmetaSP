# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from copy import deepcopy
import multiprocessing

import pandas as pd


def exec_method_on_items(method, items, use_multiprocessing=False, objects=(), *args):
    if not use_multiprocessing:
        method(items, *objects, *args)
    else:
        cpu_cores = multiprocessing.cpu_count()
        if len(items) < cpu_cores:
            cpu_cores = len(items)

        item_splits = [items[i::cpu_cores] for i in range(cpu_cores)]

        processes = []
        print(f"Spawning {cpu_cores} processes to process {len(items)} items...")
        try:
            for split in item_splits:
                object_copies = [deepcopy(obj) for obj in objects]
                p = multiprocessing.Process(
                    target=method,
                    args=(split, *object_copies, *args),
                )
                p.start()
                print(f"Process {p.pid} spawned...")
                processes.append(p)
        except KeyboardInterrupt:
            for p in processes:
                p.terminate()

        for p in processes:
            p.join()


def init_runs_log_dict():
    runs_log_dict = {
        "task_id": [],
        "run_id": [],
        "flow_id": [],
        "flow_name": [],
        "exekg_name": [],
        "elapsed_time": [],
        "error": [],
    }

    return runs_log_dict


def update_runs_log_dict(
    runs_log_dict,
    task_id,
    run_id,
    flow_id,
    flow_name,
    exekg_name,
    elapsed_time,
    error,
):
    runs_log_dict["task_id"].append(task_id)
    runs_log_dict["run_id"].append(run_id)
    runs_log_dict["flow_id"].append(flow_id)
    runs_log_dict["flow_name"].append(flow_name)
    runs_log_dict["exekg_name"].append(exekg_name)
    runs_log_dict["elapsed_time"].append(elapsed_time)
    runs_log_dict["error"].append(error)


def log_new_runs(log_path, new_run_infos_dict, lock=None):
    if lock:
        lock.acquire()

    try:
        if not log_path.exists():
            new_runs_log_df = pd.DataFrame(new_run_infos_dict)
            new_runs_log_df.to_csv(log_path, index=False)
            return

        runs_log_df = pd.read_csv(log_path)
        runs_log_extended_df = pd.concat(
            [
                runs_log_df,
                pd.DataFrame(
                    new_run_infos_dict,
                ),
            ]
        )
        runs_log_extended_df.to_csv(log_path, index=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        print(f"Error while parsing the runs log file: {e}")
    finally:
        if lock:
            lock.release()


def get_logged_run_ids(log_path):
    runs_log_df = pd.read_csv(log_path)
    return runs_log_df["run_id"].astype(int).tolist()


def init_tasks_log_dict():
    base_log_dict = {
        "task_id": [],
        "dataset_id": [],
        "dataset_name": [],
        "dataset_description": [],
        "exekg_train_task_type": [],
        "metric_to_sort_runs": [],
        "feature_data_entities": [],
        "label_data_entity": [],
        "est_procedure": [],
        "est_procedure_params": [],
        "elapsed_time": [],
        "error": [],
        "num_of_exekgs": [],
        "num_of_flows": [],
    }

    return base_log_dict


def update_tasks_log_dict(
    task_infos_dict,
    task_id,
    dataset=None,
    ml_task_type=None,
    metric_name=None,
    feature_data_entities=None,
    label_data_entity=None,
    est_procedure_sklearn=None,
    est_procedure_params_dict=None,
    elapsed_time=None,
    error=None,
):
    task_infos_dict["task_id"].append(task_id)
    task_infos_dict["dataset_id"].append(dataset.id if dataset else None)
    task_infos_dict["dataset_name"].append(dataset.name if dataset else None)
    task_infos_dict["dataset_description"].append(
        "\\n".join(dataset.description.splitlines())
        if dataset and dataset.description
        else None
    )
    task_infos_dict["exekg_train_task_type"].append(ml_task_type)
    task_infos_dict["metric_to_sort_runs"].append(metric_name)
    task_infos_dict["feature_data_entities"].append(
        ", ".join([entity.name for entity in feature_data_entities])
        if feature_data_entities
        else None
    )
    task_infos_dict["label_data_entity"].append(
        label_data_entity.name if label_data_entity else None
    )
    task_infos_dict["est_procedure"].append(est_procedure_sklearn)
    task_infos_dict["est_procedure_params"].append(est_procedure_params_dict)
    task_infos_dict["elapsed_time"].append(elapsed_time)
    task_infos_dict["error"].append(error)


def log_tasks(log_path, runs_log_path, task_infos_dict, lock):
    nums_of_exekgs = []
    nums_of_flows = []

    if lock:
        lock.acquire()

    try:
        if runs_log_path.exists():
            runs_df = pd.read_csv(runs_log_path)

            for task_id in task_infos_dict["task_id"]:
                task_runs_df = runs_df[runs_df["task_id"] == task_id]
                if task_runs_df.empty:
                    nums_of_exekgs.append(0)
                    nums_of_flows.append(0)
                    continue

                nums_of_exekgs.append(task_runs_df["error"].isnull().sum())
                nums_of_flows.append(task_runs_df["flow_id"].nunique())
        else:
            nums_of_exekgs = [0] * len(task_infos_dict["task_id"])
            nums_of_flows = [0] * len(task_infos_dict["task_id"])

        task_infos_dict["num_of_exekgs"] = nums_of_exekgs
        task_infos_dict["num_of_flows"] = nums_of_flows

        if not log_path.exists():
            task_log_df = pd.DataFrame(task_infos_dict)
            task_log_df.to_csv(log_path, index=False)
            return

        tasks_log_df = pd.read_csv(log_path)
        tasks_log_extended_df = pd.concat(
            [
                tasks_log_df,
                pd.DataFrame(
                    task_infos_dict,
                ),
            ]
        )
        tasks_log_extended_df.to_csv(log_path, index=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        print(f"Error while parsing the log file: {e}")
    finally:
        if lock:
            lock.release()
