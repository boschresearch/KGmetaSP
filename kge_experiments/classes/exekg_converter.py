# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import glob
import os
import re
import shutil
from rdflib import Graph
from concurrent.futures import ProcessPoolExecutor

from kge_experiments.config import EXEKGS_NT_PATH, EXEKGS_RAW_DIR
from kge_experiments.utils.graph_utils import DS
from kge_experiments.utils.openml_utils import read_task_info_from_logs
from kge_experiments.utils.string_utils import get_openml_ids_from_file_path


class ExeKGConverter:
    def __init__(
        self,
        filter_by_dataset_ids=None,
        exclude_flows_per_task=None,
        exclude_performance_values=False,
    ):
        self.remove_performance_values = exclude_performance_values
        self.pattern = str(EXEKGS_RAW_DIR / "task_*" / "flow_*" / "run_*.ttl")
        self.files = glob.glob(self.pattern)

        if self.remove_performance_values:
            print("Excluding performance values from the output")

        filter_by_task_ids = None
        if filter_by_dataset_ids:
            print(f"Keeping {len(filter_by_dataset_ids)} OpenML datasets ...")
            datasets_df = read_task_info_from_logs(
                filter_by_dataset_ids=filter_by_dataset_ids
            )

            filter_by_task_ids = datasets_df["task_id"].tolist()

        if exclude_flows_per_task:
            print(f"Excluding flows for {len(exclude_flows_per_task)} OpenML tasks ...")

        if filter_by_dataset_ids or exclude_flows_per_task:
            num_files_old = len(self.files)
            for file in self.files:
                task_id, flow_id, _ = get_openml_ids_from_file_path(file)
                if filter_by_dataset_ids and task_id not in filter_by_task_ids:
                    self.files.remove(file)
                    continue

                if exclude_flows_per_task and (
                    task_id in exclude_flows_per_task
                    and flow_id in exclude_flows_per_task[task_id]
                ):
                    self.files.remove(file)

            print(f"Remained {len(self.files)} of {num_files_old} runs after filtering")

    def convert_ttl_to_nt(self, file):
        output_path = file.replace(".ttl", ".nt")
        if os.path.exists(output_path):
            os.remove(output_path)
        g = Graph()
        g.parse(file, format="turtle")

        if self.remove_performance_values:
            for s, p, o in g.triples((None, DS.hasValue, None)):
                g.remove((s, p, o))

        g.serialize(destination=output_path, format="nt", encoding="utf-8")
        return output_path

    def run(self):
        if EXEKGS_NT_PATH.exists():
            EXEKGS_NT_PATH.unlink()

        nt_files = []
        num_workers = min(os.cpu_count(), 10)
        print(
            f"Converting {len(self.files)} files to .nt format using {num_workers} workers"
        )
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for i, output_path in enumerate(
                executor.map(self.convert_ttl_to_nt, self.files)
            ):
                nt_files.append(output_path)
                if (i + 1) % 100 == 0:
                    print(f"{i + 1}/{len(self.files)} files processed")
        print("All ExeKGs have been converted to .nt format")

        with open(EXEKGS_NT_PATH, "a", encoding="utf-8") as concatenated_file:
            print("Concatenating all .nt files into a single file...")
            for nt_file in nt_files:
                with open(nt_file, "r", encoding="utf-8") as f:
                    shutil.copyfileobj(f, concatenated_file)
        print(f"All ExeKGs have been concatenated into {EXEKGS_NT_PATH}")

        print("Cleaning up temporary .nt files ...")
        for nt_file in nt_files:
            os.remove(nt_file)
