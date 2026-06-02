# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from multiprocessing import Lock
import time
import pandas as pd
from openml_exekgs_generation.config import (
    EXEKGS_OUTPUT_DIR,
    LOG_DIR,
    METRIC_FOR_SORTING_PIPELINES_DICT,
    RUNS_LOG_FILENAME,
    TASKS_LOG_FILENAME,
    TASKS_PATH,
)
from openml_exekgs_generation.utils.exec_utils import (
    exec_method_on_items,
    get_logged_run_ids,
    init_tasks_log_dict,
    log_tasks,
    update_tasks_log_dict,
)
from openml_exekgs_generation.utils.exekg_creation_utils import (
    create_exekgs_from_runs,
    init_exekg,
)
from openml_exekgs_generation.utils.openml_utils import (
    get_best_runs_for_task,
    get_flows,
    get_task_runs_and_info,
    get_tasks,
)
from exe_kg_lib import ExeKGEditor
from exe_kg_lib.utils.query_utils import NoResultsError


class OpenMLCrawler:
    def __init__(
        self,
        multiprocess_runs=False,
        multiprocess_tasks=False,
        task_ids=None,
        use_processed_tasks_and_runs=False,
        req_offset_step=1000,
        num_tasks_to_process=10,
        num_runs_per_flow_of_task=10,
    ):
        self.exekgs_output_dir = EXEKGS_OUTPUT_DIR
        self.log_dir = LOG_DIR
        self.runs_log_filename = RUNS_LOG_FILENAME
        self.tasks_log_filename = TASKS_LOG_FILENAME

        self.runs_log_path = self.log_dir / self.runs_log_filename
        self.tasks_log_path = self.log_dir / self.tasks_log_filename

        self.multiprocess_runs = multiprocess_runs
        self.multiprocess_tasks = multiprocess_tasks
        self.task_ids = task_ids
        self.use_processed_tasks_and_runs = use_processed_tasks_and_runs
        self.req_offset_step = req_offset_step
        self.num_tasks_to_process = num_tasks_to_process
        self.num_runs_per_flow_of_task = num_runs_per_flow_of_task

    def initialize_output_directories(self):
        self.exekgs_output_dir.mkdir(parents=True, exist_ok=True)
        self.runs_log_path.parents[0].mkdir(parents=True, exist_ok=True)
        self.tasks_log_path.parents[0].mkdir(parents=True, exist_ok=True)

    def get_task_ids(self, offset=0, num_tasks_to_process=10):
        tasks = get_tasks(num_tasks_to_process, offset)
        task_ids = tasks["tid"].tolist()
        task_types = tasks["task_type"].tolist()

        task_ids = [int(task_id) for task_id in task_ids]
        print(f"Found {len(task_ids)} tasks in total.")
        return task_ids, task_types

    def get_sklearn_flows(self):
        sklearn_flows = get_flows(r"^sklearn")
        sklearn_flow_ids = sklearn_flows["id"].astype(int).tolist()
        print(f"Found {len(sklearn_flows)} sklearn flows.")
        return sklearn_flow_ids

    def load_processed_tasks(self):
        processed_task_infos_dict = init_tasks_log_dict()
        if self.tasks_log_path.exists():
            processed_task_infos_dict = pd.read_csv(self.tasks_log_path).to_dict(
                "list"
            )
        else:
            self.tasks_log_path.parents[0].mkdir(parents=True, exist_ok=True)
        return processed_task_infos_dict, self.tasks_log_path

    def get_processed_runs(self):
        processed_run_ids = []
        if self.runs_log_path.exists():
            processed_run_ids = get_logged_run_ids(self.runs_log_path)
        return processed_run_ids

    def process_task(
        self,
        task_id,
        sklearn_flow_ids,
        exe_kg,
        processed_task_infos_dict,
        processed_run_ids,
        task_i,
        task_ids,
        lock,
    ):
        start = time.time()
        try:
            (
                run_ids_metric_values,
                metric_name,
                dataset,
                feature_data_entities,
                label_data_entity,
                est_procedure_sklearn,
                est_procedure_params_dict,
                ml_task_type,
            ) = get_task_runs_and_info(
                task_id,
                filter_flow_ids=sklearn_flow_ids,
                num_runs_per_flow=self.num_runs_per_flow_of_task,
            )
        except (ValueError, NotImplementedError) as e:
            print(f"{e} Skipping...")
            update_tasks_log_dict(
                processed_task_infos_dict,
                task_id,
                elapsed_time=time.time() - start,
                error=str(e),
            )
            return False

        exe_kg.clear_created_kg()

        try:
            train_x, train_real_y, test_x, test_real_y = init_exekg(
                exe_kg,
                dataset,
                feature_data_entities,
                label_data_entity,
                est_procedure_sklearn,
                est_procedure_params_dict,
            )
        except NoResultsError as e:
            print(f"{e} Skipping...")
            return False

        if not run_ids_metric_values:
            error = f"No sklearn-based runs found for task {task_id}."
            print(f"{error} Skipping...")
            update_tasks_log_dict(
                processed_task_infos_dict,
                task_id,
                dataset,
                ml_task_type,
                metric_name,
                feature_data_entities,
                label_data_entity,
                est_procedure_sklearn,
                est_procedure_params_dict,
                time.time() - start,
                str(error),
            )
            return False

        if not self.use_processed_tasks_and_runs:
            run_ids_metric_values = list(
                filter(lambda x: x[0] not in processed_run_ids, run_ids_metric_values)
            )
            print(
                f"Skipping {len(processed_run_ids)} runs that have already been processed..."
            )

        exec_method_on_items(
            create_exekgs_from_runs,
            run_ids_metric_values,
            self.multiprocess_runs,
            (exe_kg, train_x, train_real_y, test_x, test_real_y),
            task_id,
            ml_task_type,
            lock,
        )

        update_tasks_log_dict(
            processed_task_infos_dict,
            task_id,
            dataset,
            ml_task_type,
            metric_name,
            feature_data_entities,
            label_data_entity,
            est_procedure_sklearn,
            est_procedure_params_dict,
            time.time() - start,
            None,
        )
        return True

    def crawl_runs_of_tasks(
        self,
        task_ids,
        exe_kg,
        sklearn_flow_ids,
        processed_run_ids,
        lock_for_tasks,
        lock_for_runs,
    ):
        task_infos_dict = init_tasks_log_dict()
        for task_i, task_id in enumerate(task_ids):
            success = self.process_task(
                task_id,
                sklearn_flow_ids,
                exe_kg,
                task_infos_dict,
                processed_run_ids,
                task_i,
                task_ids,
                lock_for_runs,
            )

            if task_i % 5 == 0 or task_i == len(task_ids) - 1:
                print(f"Processed {task_i + 1} task(s) out of {len(task_ids)}.")
                log_tasks(
                    self.tasks_log_path,
                    self.runs_log_path,
                    task_infos_dict,
                    lock_for_tasks,
                )
                task_infos_dict = init_tasks_log_dict()

    def crawl_runs(self):
        self.initialize_output_directories()
        processed_task_infos_dict, _ = self.load_processed_tasks()
        processed_task_ids = [
            int(task_id) for task_id in processed_task_infos_dict["task_id"]
        ]
        processed_run_ids = self.get_processed_runs()
        sklearn_flow_ids = self.get_sklearn_flows()
        exe_kg = ExeKGEditor()
        if self.task_ids is not None:
            self.num_tasks_to_process = len(self.task_ids)

        for offset in range(0, self.num_tasks_to_process, self.req_offset_step):
            TASKS_PATH.unlink(missing_ok=True)

            print(
                f"Processing tasks {offset} to {offset + self.req_offset_step} from {self.num_tasks_to_process}..."
            )
            if self.use_processed_tasks_and_runs:
                if offset >= len(processed_task_ids):
                    break
                task_ids = processed_task_ids[offset : offset + self.req_offset_step]
            else:
                if self.task_ids is None:
                    task_ids, _ = self.get_task_ids(
                        offset, self.req_offset_step
                    )
                else:
                    task_ids = self.task_ids[offset : offset + self.req_offset_step]

                num_tasks_init = len(task_ids)
                task_ids = list(
                    filter(lambda task_id: task_id not in processed_task_ids, task_ids)
                )

                print(
                    f"Skipping {num_tasks_init - len(task_ids)} tasks that have already been processed..."
                )

            exec_method_on_items(
                self.crawl_runs_of_tasks,
                task_ids,
                self.multiprocess_tasks,
                (exe_kg,),
                sklearn_flow_ids,
                processed_run_ids,
                Lock() if self.multiprocess_tasks else None,
                Lock() if self.multiprocess_runs else None,
            )
