# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0


import ast
import os
import pandas as pd
from kge_experiments.config import (
    EXEKGS_RAW_DIR,
    OUTPUT_DIR,
    RDF2VEC_MODEL_DIR,
    SIMILARITIES_CSV_PATH,
)

from kge_experiments.classes.exekg_dataset import ExeKGDataset as RDF2VecKG

from kge_experiments.utils.model_utils import (
    load_rdf2vec_model,
)
from kge_experiments.utils.embedding_utils import (
    get_dataset_emb,
    get_dataset_info,
)
from kge_experiments.utils.openml_utils import (
    create_dataset_to_task_ids,
    get_flow_name,
    read_task_info_from_logs,
)
from kge_experiments.utils.similarity_utils import (
    calculate_emb_similarities,
    get_nums_of_common_items,
)
from kge_experiments.utils.ged_utils import (
    calculate_geds_for_dataset_combinations,
)
from kge_experiments.utils.openml_utils import statistics_on_openml_tasks


class DatasetSimilarityCalculator:
    def __init__(
        self,
        verbose=False,
        rdf2vec_walk_distance=10,
        rdf2vec_num_walks=10,
        rdf2vec_walk_strategy="random",
        use_mlseakg=False,
        use_mkga=False,
        filter_by_dataset_ids=None,
        exclude_flows_per_task=None,
    ):
        self.verbose = verbose
        self._rdf2vec_kg = None
        self._rdf2vec_model = None
        self.datasets_df = read_task_info_from_logs(
            filter_by_dataset_ids=filter_by_dataset_ids
        )
        if verbose:
            statistics_on_openml_tasks(self.datasets_df)
        self.dataset_to_task_ids = create_dataset_to_task_ids(self.datasets_df)
        if len(self.dataset_to_task_ids) < len(self.datasets_df):
            print(
                f"Number of datasets with multiple tasks: {len(list(filter(lambda x: len(x) > 1, self.dataset_to_task_ids.values())))}"
            )

        self.rdf2vec_walk_distance = rdf2vec_walk_distance
        self.rdf2vec_num_walks = rdf2vec_num_walks
        self.rdf2vec_walk_strategy = rdf2vec_walk_strategy

        self.use_mlseakg = use_mlseakg
        self.use_mkga = use_mkga

        self.rdf2vec_model_suffix = (
            str(RDF2VEC_MODEL_DIR)
            .split(os.path.sep)[-1]
            .format(
                d=rdf2vec_walk_distance,
                w=rdf2vec_num_walks,
                ws=rdf2vec_walk_strategy,
                kg=("exekgs+mlseakg" if use_mlseakg else "exekgs")
                + ("_mkga" if use_mkga else ""),
            )
        )
        self.exclude_flows_per_task = exclude_flows_per_task

    @property
    def rdf2vec_kg(self):
        if self._rdf2vec_kg is None:
            self._rdf2vec_kg = RDF2VecKG(self.use_mlseakg, self.use_mkga)
        return self._rdf2vec_kg

    @property
    def rdf2vec_model(self):
        if self._rdf2vec_model is None:
            self._rdf2vec_model = load_rdf2vec_model(
                self.rdf2vec_walk_distance,
                self.rdf2vec_num_walks,
                self.rdf2vec_walk_strategy,
                self.use_mlseakg,
                self.use_mkga,
            )
        return self._rdf2vec_model

    def calculate_similarities(
        self,
        kge_type,
        calc_data_entity=True,
        calc_data_entity_and_pipeline=True,
        calc_pipeline=True,
    ):
        results = {}
        configs = [
            {
                "flag": calc_data_entity,
                "use_data_entity_embeddings": True,
                "use_pipeline_embeddings": False,
                "result_key_suffix": f"_data_entity_emb_{kge_type}",
            },
            {
                "flag": calc_data_entity_and_pipeline,
                "use_data_entity_embeddings": True,
                "use_pipeline_embeddings": True,
                "result_key_suffix": f"_data_entity_emb+pipeline_emb_{kge_type}",
            },
            {
                "flag": calc_pipeline,
                "use_data_entity_embeddings": False,
                "use_pipeline_embeddings": True,
                "result_key_suffix": f"_pipeline_emb_{kge_type}",
            },
        ]

        stored_pipeline_emb_rdf2vec_dict = {}
        stored_data_entity_emb_rdf2vec_dict = {}
        for config in configs:
            if not config["flag"]:
                continue

            avg_embeddings = get_dataset_emb(
                self.dataset_to_task_ids,
                self.rdf2vec_kg,
                self.rdf2vec_model,
                self.datasets_df,
                EXEKGS_RAW_DIR,
                kge_type,
                stored_pipeline_emb_rdf2vec_dict,
                stored_data_entity_emb_rdf2vec_dict,
                use_data_entity_embeddings=config["use_data_entity_embeddings"],
                use_pipeline_embeddings=config["use_pipeline_embeddings"],
                exclude_flows_per_task=self.exclude_flows_per_task,
            )
            cosine_sims, euclidean_dist_sims, manhattan_dist_sims, dot_product_sims = (
                calculate_emb_similarities(avg_embeddings)
            )
            for similarities, metric in zip(
                [
                    cosine_sims,
                    euclidean_dist_sims,
                    manhattan_dist_sims,
                    dot_product_sims,
                ],
                ["cosine", "euclidean", "manhattan", "dot_product"],
            ):
                result_key = f"{metric}{config['result_key_suffix']}"
                df_similarities = pd.DataFrame(
                    similarities,
                    columns=[
                        "Dataset 1",
                        "Dataset 2",
                        result_key,
                    ],
                )
                results[result_key] = df_similarities

        return results

    def get_dataset_info_dfs(self):
        columns, nums_of_columns, flows, nums_of_flows = get_dataset_info(
            self.datasets_df, EXEKGS_RAW_DIR, self.exclude_flows_per_task
        )

        nums_of_common_columns = get_nums_of_common_items(columns)

        df_nums_of_common_columns = pd.DataFrame(
            nums_of_common_columns,
            columns=["Dataset 1", "Dataset 2", "Number of Common Columns"],
        )

        df_nums_of_columns = pd.DataFrame(
            list(nums_of_columns.items()),
            columns=["Dataset", "Number of Columns"],
        )

        nums_of_common_flows = get_nums_of_common_items(flows)

        df_nums_of_common_flows = pd.DataFrame(
            nums_of_common_flows,
            columns=["Dataset 1", "Dataset 2", "Number of Common Flows"],
        )

        df_nums_of_flows = pd.DataFrame(
            list(nums_of_flows.items()),
            columns=["Dataset", "Number of Flows"],
        )

        df_flows = pd.DataFrame(
            list(flows.items()),
            columns=["Dataset", "Flows"],
        )

        return {
            "Number of Common Columns": df_nums_of_common_columns,
            "Number of Columns": df_nums_of_columns,
            "Number of Common Flows": df_nums_of_common_flows,
            "Number of Flows": df_nums_of_flows,
            "Flows": df_flows,
        }

    def get_ged_df(self, dataset_pairs, similarities_df, num_processes, chunk_size):
        print(f"Calculating graph edit distances for {len(dataset_pairs)} ...")

        geds = calculate_geds_for_dataset_combinations(
            dataset_pairs,
            similarities_df,
            self.dataset_to_task_ids,
            self.exclude_flows_per_task,
            num_processes,
            chunk_size,
        )

        if len(geds) == 0:
            return {}

        df_graph_edit_distances = pd.DataFrame(
            geds,
            columns=[
                "Dataset 1",
                "Dataset 2",
                "Avg Pairwise GED",
            ],
        )

        return {"Avg Pairwise GED": df_graph_edit_distances}

    def calculate_and_add_geds_to_similarities(
        self,
        # sample_n_for_geds=10000,
        replace_geds=False,
        add_dataset_info=False,
        replace_info=False,
        num_processes=10,
        chunk_size=5,
    ):
        if not SIMILARITIES_CSV_PATH.exists():
            print(
                'No similarities file found. First calculate KGE-based similarities using "calculate-kge-similarities" command.'
            )
            return

        merged_df = pd.read_csv(SIMILARITIES_CSV_PATH)

        if add_dataset_info:
            merged_df = self.add_dataset_info(replace_info, merged_df)

        dataset_pairs = merged_df[["Dataset 1", "Dataset 2"]].values.tolist()

        ged_dfs_dict = self.get_ged_df(
            dataset_pairs, merged_df, num_processes, chunk_size
        )
        if len(ged_dfs_dict) == 0:
            return

        list(ged_dfs_dict.values())[0].to_csv(OUTPUT_DIR / "geds.csv", index=False)

        if replace_geds:
            merged_df = merged_df.drop(
                columns=[list(ged_dfs_dict.keys())[0]], errors="ignore"
            )
        else:
            merged_df = merged_df.merge(
                list(ged_dfs_dict.values())[0],
                on=["Dataset 1", "Dataset 2"],
                how="left",
            )

            merged_df["Avg Pairwise GED_x"] = merged_df["Avg Pairwise GED_x"].fillna(
                merged_df["Avg Pairwise GED_y"]
            )

            merged_df = merged_df.drop(columns=["Avg Pairwise GED_y"])
            merged_df = merged_df.rename(
                columns={"Avg Pairwise GED_x": "Avg Pairwise GED"}
            )

        self.backup_and_save(merged_df)

    def calculate_kge_similarities(
        self,
        replace_similarities=False,
        replace_info=False,
        add_data_entity_sim=False,
        add_data_entity_and_pipeline_sim=False,
        add_pipeline_sim=False,
        add_dataset_info=False,
    ):

        kge_type = "rdf2vec_" + self.rdf2vec_model_suffix

        similarities_dict = self.calculate_similarities(
            kge_type=kge_type,
            calc_data_entity=add_data_entity_sim,
            calc_data_entity_and_pipeline=add_data_entity_and_pipeline_sim,
            calc_pipeline=add_pipeline_sim,
        )

        if SIMILARITIES_CSV_PATH.exists():
            merged_df = pd.read_csv(SIMILARITIES_CSV_PATH)
        else:
            merged_df = None

        if add_dataset_info:
            merged_df = self.add_dataset_info(replace_info, merged_df)

        for col, df in similarities_dict.items():
            if merged_df is None:
                merged_df = df
                continue

            if replace_similarities:
                merged_df = merged_df.drop(columns=[col], errors="ignore")
            merged_df = merged_df.merge(df, on=["Dataset 1", "Dataset 2"])

        self.backup_and_save(merged_df)

    def backup_and_save(self, merged_df):
        old_csv_path = str(SIMILARITIES_CSV_PATH).replace(".csv", "_old.csv")
        if os.path.exists(old_csv_path):
            os.remove(old_csv_path)
        if SIMILARITIES_CSV_PATH.exists():
            os.rename(SIMILARITIES_CSV_PATH, old_csv_path)

        merged_df.to_csv(SIMILARITIES_CSV_PATH, index=False)
        print(
            f"Similarities added to {SIMILARITIES_CSV_PATH} and the old file is renamed to {old_csv_path}"
        )

    def add_dataset_info(self, replace_info, merged_df):
        dataset_info_dict = self.get_dataset_info_dfs()
        for col, df in dataset_info_dict.items():
            if merged_df is None:
                merged_df = df
                continue

            if col == "Number of Common Columns" or col == "Number of Common Flows":
                if replace_info:
                    merged_df = merged_df.drop(columns=[col], errors="ignore")

                merged_df = merged_df.merge(df, on=["Dataset 1", "Dataset 2"])
            else:
                if replace_info:
                    merged_df = merged_df.drop(
                        columns=[col + " 1", col + " 2"], errors="ignore"
                    )
                merged_df = (
                    merged_df.merge(df, left_on="Dataset 1", right_on="Dataset")
                    .merge(df, left_on="Dataset 2", right_on="Dataset")
                    .drop(["Dataset_x", "Dataset_y"], axis=1)
                )

                merged_df = merged_df.rename(
                    columns={
                        col + "_x": col + " 1",
                        col + "_y": col + " 2",
                    },
                )

        return merged_df

    def process_csv_for_manual_check(
        self,
        sort_by="Similarity_pipeline_emb",
        keep_first_n=10,
        csv_suffix="",
    ):
        csv_path = str(SIMILARITIES_CSV_PATH).replace(".csv", f"{csv_suffix}.csv")
        existing_df = pd.read_csv(csv_path)
        existing_df_sorted = existing_df.sort_values(
            by=sort_by, ascending=False
        ).reset_index(drop=True)
        existing_df_sorted = pd.concat(
            [
                existing_df_sorted.head(keep_first_n),
                existing_df_sorted.tail(keep_first_n),
            ]
        )

        existing_df_sorted["Flow Code 1"] = existing_df_sorted["Flows 1"].apply(
            lambda flow_ids_s: list(
                set(
                    [get_flow_name(flow_id) for flow_id in ast.literal_eval(flow_ids_s)]
                )
            )[0]
        )
        existing_df_sorted["Flow Code 2"] = existing_df_sorted["Flows 2"].apply(
            lambda flow_ids_s: list(
                set(
                    [get_flow_name(flow_id) for flow_id in ast.literal_eval(flow_ids_s)]
                )
            )[0]
        )

        new_csv_path = str(csv_path).replace(".csv", f"_sort_by_{sort_by}.csv")
        existing_df_sorted.to_csv(new_csv_path, index=False)
        print(f"Similarities saved to {new_csv_path}")
