# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
import pathlib
import pickle
import pandas as pd
import numpy as np
import json
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV, GroupKFold, train_test_split
from sklearn.metrics import (
    classification_report,
    f1_score,
    make_scorer,
    mean_squared_error,
    r2_score,
    accuracy_score,
)
from typing import Tuple, Optional, Dict, Any
from tqdm import tqdm
from kge_experiments.config import DATA_DIR, EXEKGS_RAW_DIR, PERF_PRED_MODEL_CONFIGS
from kge_experiments.constants.filter_by_datasets import FILTER_BY_DATASET_IDS
from kge_experiments.utils.embedding_utils import (
    get_data_entity_emb_rdf2vec,
    get_data_entity_emb_pykeen_lp,
)
from kge_experiments.classes.exekg_dataset import ExeKGDataset as RDF2VecKG
from kge_experiments.classes.exekg_pykeen_dataset import ExeKGPykeenDataset
from kge_experiments.utils.model_utils import load_rdf2vec_model, load_pykeen_lp_model
from kge_experiments.utils.openml_utils import read_task_info_from_logs
from kge_experiments.utils.string_utils import get_openml_ids_from_file_path
from sklearn.metrics.pairwise import cosine_distances

PIPELINE_EVALUATIONS_PATH = DATA_DIR / "pipeline_evaluations.json"
RUN_TO_TASK_PATH = DATA_DIR / "run2task.json"
REMOTE_RESULTS_DIR = pathlib.Path(
    "/fs/scratch/rb_bd_dlp_rng-dl01_cr_AIQ_employees/students/klr2rng/perf_pred_results"
)

PIPELINE_EMBEDDINGS_PICKLE_PATH = (
    DATA_DIR
    / "pipeline_embeddings_cache_d_{walk_distance}_w_{num_walks}_mlseakg_{use_mlseakg}_mkga_{use_mkga}.pkl"
)
DATASET_EMBEDDINGS_PICKLE_PATH = (
    DATA_DIR
    / "dataset_embeddings_cache_d_{walk_distance}_w_{num_walks}_mlseakg_{use_mlseakg}_mkga_{use_mkga}.pkl"
)
PYKEEN_LP_PIPELINE_EMBEDDINGS_PICKLE_PATH = (
    DATA_DIR
    / "pipeline_embeddings_cache_pykeen_lp_{model_name}_mlseakg_{use_mlseakg}_mkga_{use_mkga}.pkl"
)
PYKEEN_LP_DATASET_EMBEDDINGS_PICKLE_PATH = (
    DATA_DIR
    / "dataset_embeddings_cache_pykeen_lp_{model_name}_mlseakg_{use_mlseakg}_mkga_{use_mkga}.pkl"
)

RESULTS_DIR_DATASET = (
    REMOTE_RESULTS_DIR
    / "results_target_{target}_model_{model}_emb_aggr_{emb_aggr_type}_dataset_emb_{dataset_emb_source}_pipeline_emb_{pipeline_emb_source}{pykeen_lp_model_part}_mlseakg_{use_mlseakg}_mkga_{use_mkga}_split_dataset_mintrainsamples_{min_train_samples_per_run_id}"
)
RESULTS_DIR_PIPELINE = (
    REMOTE_RESULTS_DIR
    / "results_target_{target}_model_{model}_emb_aggr_{emb_aggr_type}_dataset_emb_{dataset_emb_source}_pipeline_emb_{pipeline_emb_source}{pykeen_lp_model_part}_mlseakg_{use_mlseakg}_mkga_{use_mkga}_split_pipeline_mintrainsamples_{min_train_samples_per_run_id}"
)

METAFEATURE_PATHS = {
    "metafeatures_all": DATA_DIR / "metafeatures_all.pkl",
    "metafeatures_mlsea": DATA_DIR / "metafeatures_mlsea.pkl",
    "metafeatures_information_theory": DATA_DIR / "metafeatures_information_theory.pkl",
    "metafeatures_landmarkers": DATA_DIR / "metafeatures_landmarkers.pkl",
    "metafeatures_simple": DATA_DIR / "metafeatures_simple.pkl",
    "metafeatures_statistical": DATA_DIR / "metafeatures_statistical.pkl",
}

N_FOLDS = 10


class PipelinePerformancePredictor:
    """Class for pipeline performance prediction using embeddings and machine learning."""

    def __init__(
        self,
        split_mode: str = "dataset",
        target: str = "f1_score",
        emb_aggr_type: str = "concat",
        model: str = "RF",
        dataset_emb_source: str = "rdf2vec",
        pipeline_emb_source: str = "rdf2vec",
        use_mlseakg: bool = False,
        use_mkga: bool = False,
        walk_distance: int = 20,
        num_walks: int = 10,
        min_train_samples_per_run_id: int = 50,
        verbose: bool = True,
        pykeen_lp_model_path: Optional[str] = None,
    ):
        """Initialize the pipeline performance prediction class."""
        self.split_mode = split_mode
        self.target = target
        self.emb_aggr_type = emb_aggr_type
        self.model = model
        self.dataset_emb_source = dataset_emb_source
        self.pipeline_emb_source = pipeline_emb_source
        self.use_mlseakg = use_mlseakg
        self.use_mkga = use_mkga
        self.walk_distance = walk_distance
        self.num_walks = num_walks
        self.min_train_samples_per_run_id = min_train_samples_per_run_id
        self.verbose = verbose
        self.pykeen_lp_model_path = pykeen_lp_model_path

        # Initialize attributes
        self.rdf2vec_kg = self.rdf2vec_model = self.data_df = None
        self.pykeen_lp_model = self.pykeen_lp_dataset = None
        self.openml_datasets_df = self.run_to_task_dict = None

        if self.verbose:
            print(f"Initialized PipelinePerformancePrediction with:")
            print(f"Split mode: {split_mode}, Target: {target}")
            print(f"Embedding aggregation: {emb_aggr_type}, Model: {model}")
            print(f"Dataset embedding source: {self.dataset_emb_source}")
            print(f"Pipeline embedding source: {self.pipeline_emb_source}")
            print(f"use_mlseakg: {use_mlseakg}, use_mkga: {use_mkga}")
            print(f"min_train_samples_per_run_id: {min_train_samples_per_run_id}")

    def _get_pipeline_name_from_run_id_and_task_id(
        self, run_id: int, task_id: int, exekgs_path: pathlib.Path
    ) -> Optional[str]:
        """Create a pipeline name of the format pipeline_task_{task_id}_flow_{flow_id}_run_{run_id}."""
        path_pattern = str(
            exekgs_path / f"task_{task_id}" / "flow_*" / f"run_{run_id}.ttl"
        )
        file_paths = glob.glob(path_pattern)
        if not file_paths:
            return None
        _, flow_id, _ = get_openml_ids_from_file_path(file_paths[0])
        return f"pipeline_task_{task_id}_flow_{flow_id}_run_{run_id}"

    def _get_pipeline_emb(self, pipeline_name: str) -> Optional[np.ndarray]:
        """Get pipeline embedding from RDF2Vec model."""
        pipeline_comp_names = [
            entity
            for entity in self.rdf2vec_kg.entities
            if entity.endswith(f"_{pipeline_name}")
            and "Concatenation" not in entity
            and "Method" in entity
        ]
        if not pipeline_comp_names:
            return None

        pipeline_comp_embeddings = []
        for name in pipeline_comp_names:
            try:
                pipeline_comp_embeddings.append(self.rdf2vec_model.wv.get_vector(name))
            except KeyError as e:
                if self.verbose:
                    print(
                        f"KeyError: {e} - Pipeline component {name} not found in rdf2vec vocabulary."
                    )
                continue

        if not pipeline_comp_embeddings:
            if self.verbose:
                print(f"Failed pipeline: {pipeline_name}")
            return None
        return np.mean(pipeline_comp_embeddings, axis=0)

    def _get_pykeen_lp_pipeline_emb(self, pipeline_name: str) -> Optional[np.ndarray]:
        """Get pipeline embedding from PyKEEN LP model."""
        import torch

        # Get the entity representation from the model
        entity_representation = self.pykeen_lp_model.entity_representations[0]
        entity_to_id = self.pykeen_lp_dataset.entity_to_id
        
        # Find all method components for this pipeline
        pipeline_comp_names = [
            entity
            for entity in entity_to_id.keys()
            if entity.endswith(f"_{pipeline_name}")
            and "Concatenation" not in entity
            and "Method" in entity
        ]
        
        if not pipeline_comp_names:
            return None
        
        pipeline_comp_embeddings = []
        for name in pipeline_comp_names:
            if name in entity_to_id:
                entity_id = entity_to_id[name]
                embedding = entity_representation(indices=torch.tensor([entity_id])).detach().cpu().numpy()[0]
                pipeline_comp_embeddings.append(embedding)
            elif self.verbose:
                print(f"Pipeline component {name} not found in PyKEEN entity mappings.")
        
        if not pipeline_comp_embeddings:
            if self.verbose:
                print(f"Failed pipeline: {pipeline_name}")
            return None
        return np.mean(pipeline_comp_embeddings, axis=0)

    def _get_dataset_embedding(self, dataset_id: int) -> Optional[np.ndarray]:
        """Get dataset embedding from RDF2Vec model."""
        dataset_info = self.openml_datasets_df[
            self.openml_datasets_df["dataset_id"] == dataset_id
        ].iloc[0]
        feature_entities = dataset_info["feature_data_entities"].split(", ")
        data_entities = feature_entities + [dataset_info["label_data_entity"]]

        data_entity_embeddings, _ = get_data_entity_emb_rdf2vec(
            data_entities, self.rdf2vec_kg, self.rdf2vec_model, 0
        )

        if not data_entity_embeddings:
            if self.verbose:
                print(f"Failed dataset: {dataset_id}")
            return None
        return np.mean(data_entity_embeddings, axis=0)

    def _fuse_embeddings(
        self,
        pipeline_embedding: np.ndarray,
        dataset_embedding: np.ndarray,
        aggr_type: str = "concat",
    ) -> np.ndarray:
        """Fuses pipeline and dataset embeddings."""
        # Convert complex embeddings to real by splitting into real/imaginary parts
        pipeline_embedding = self._convert_complex_to_real(pipeline_embedding)
        dataset_embedding = self._convert_complex_to_real(dataset_embedding)
        
        if aggr_type == "concat":
            return np.concatenate([pipeline_embedding, dataset_embedding])
        elif aggr_type == "mean":
            return np.mean([pipeline_embedding, dataset_embedding], axis=0)
        else:
            raise ValueError(f"Unknown aggregation type: {aggr_type}")

    @staticmethod
    def _convert_complex_to_real(X: np.ndarray) -> np.ndarray:
        """Convert complex-valued arrays to real-valued by splitting into real and imaginary parts.
        
        Args:
            X: Input array of shape (n_samples, n_features) which may be complex
            
        Returns:
            Real-valued array. If input was complex, shape is (n_samples, 2*n_features),
            otherwise returns the input unchanged.
        """
        if np.iscomplexobj(X):
            if X.ndim == 1:
                # Handle 1D arrays
                return np.concatenate([X.real, X.imag])
            else:
                # Handle 2D arrays - concatenate real and imaginary parts along feature axis
                return np.hstack([X.real, X.imag])
        return X

    def _bin_target_variable(
        self, data_df: pd.DataFrame, target_col: str, num_bins: int = 3
    ) -> pd.DataFrame:
        """Bin the target variable into proportional intervals using pd.qcut."""
        labels = [f"bin_{i}" for i in range(1, num_bins + 1)]
        binned_col = f"{target_col}_binned"
        data_df[binned_col] = pd.qcut(data_df[target_col], q=num_bins, labels=labels)

        if self.verbose:
            print("\nBinning Statistics:")
            bin_edges = pd.qcut(data_df[target_col], q=num_bins, retbins=True)[1]
            for i in range(len(bin_edges) - 1):
                bin_data = data_df[
                    (data_df[target_col] >= bin_edges[i])
                    & (data_df[target_col] <= bin_edges[i + 1])
                ]
                print(f"{labels[i]}:")
                print(f"  Min: {bin_edges[i]:.4f}, Max: {bin_edges[i + 1]:.4f}")
                print(f"  Count: {len(bin_data)}")
                print(f"  Mean: {bin_data[target_col].mean():.4f}")
                print(f"  Std Dev: {bin_data[target_col].std():.4f}")
                print(f"  Range: {bin_edges[i + 1] - bin_edges[i]:.4f}")
        return data_df

    def _train_test_split_no_overlap(
        self,
        data_df: pd.DataFrame,
        group_col: str,
        target_col: str,
        embedding_col: str,
        train_size: float = 0.7,
        test_size: float = 0.3,
        random_state: int = 42,
    ) -> Tuple:
        """Split data ensuring no overlap between groups in train and test sets."""
        splits_save_dir = (
            DATA_DIR
            / f"splits_{group_col}_{(1-test_size)*100:.0f}_{test_size*100:.0f}_target_{target_col}_no_overlap"
        )
        splits_save_dir.mkdir(parents=True, exist_ok=True)

        group_ids = data_df[group_col].unique()
        group_to_idx = {gid: i for i, gid in enumerate(group_ids)}
        groups = data_df[group_col].map(group_to_idx)

        X = np.stack(data_df[embedding_col].values)
        y = data_df[target_col].values

        train_ids, test_ids = train_test_split(
            group_ids, test_size=test_size, random_state=random_state
        )
        train_mask, test_mask = data_df[group_col].isin(train_ids), data_df[
            group_col
        ].isin(test_ids)

        X_train, y_train, groups_train = (
            X[train_mask],
            y[train_mask],
            groups[train_mask],
        )
        X_test, y_test, groups_test = X[test_mask], y[test_mask], groups[test_mask]

        assert (
            len(set(train_ids).intersection(set(test_ids))) == 0
        ), "Train and test sets have overlapping group IDs!"

        # Save splits
        data_df[train_mask][[group_col, "run_id", target_col]].to_csv(
            splits_save_dir / "train.csv", index=False
        )
        data_df[test_mask][[group_col, "run_id", target_col]].to_csv(
            splits_save_dir / "test.csv", index=False
        )

        if self.verbose:
            print(f"Train and test sets saved to {splits_save_dir}")
            print(f"Train: {len(X_train)}, Test: {len(X_test)}")

        return (
            X_train,
            X_test,
            pd.Series(y_train).reset_index(drop=True),
            y_test,
            pd.Series(groups_train).reset_index(drop=True),
            groups_test,
            train_mask,
            test_mask,
        )

    def _closest_embedding_baseline(
        self, X_train: np.ndarray, y_train: pd.Series, X_test: np.ndarray
    ) -> np.ndarray:
        """Baseline using closest embedding in training set."""
        dists = cosine_distances(X_test, X_train)
        closest_idxs = np.argmin(dists, axis=1)
        return y_train.iloc[closest_idxs].values

    def _average_baseline(self, y_train: pd.Series, X_test: np.ndarray) -> np.ndarray:
        """Baseline using average target value."""
        return np.full(len(X_test), y_train.mean())

    def load_data(self):
        """Load and prepare the data for training."""
        # Load OpenML datasets info
        self.openml_datasets_df = read_task_info_from_logs(
            filter_by_dataset_ids=FILTER_BY_DATASET_IDS
        )
        dataset_ids = self.openml_datasets_df["dataset_id"].unique()

        # Load run to task mapping and pipeline evaluations
        with open(RUN_TO_TASK_PATH, "r") as f:
            self.run_to_task_dict = json.load(f)
        with open(PIPELINE_EVALUATIONS_PATH, "r") as f:
            data_json = json.load(f)

        # Flatten JSON structure
        data = []
        for dataset_id, runs in data_json.items():
            dataset_id = int(dataset_id)
            if dataset_id not in dataset_ids:
                if self.verbose:
                    print(f"Dataset ID {dataset_id} not found in dataset_to_task_ids")
                continue
            for run_id, run_info in runs.items():
                if (
                    run_info.get("run_success", False)
                    and "test_f1" in run_info["run_metrics"]
                ):
                    data.append(
                        {
                            "dataset_id": dataset_id,
                            "run_id": int(run_id),
                            "f1_score": run_info["run_metrics"]["test_f1"],
                            "accuracy": run_info["run_metrics"]["test_accuracy"],
                            "precision": run_info["run_metrics"]["test_precision"],
                            "fit_time": run_info["run_metrics"]["fit_time"],
                        }
                    )

        self.data_df = pd.DataFrame(data)
        if self.verbose:
            print(f"Loaded {len(self.data_df)} pipeline evaluations")

    def load_models(self):
        """Load RDF2Vec or PyKEEN LP model and knowledge graph if needed."""
        # Load PyKEEN LP model if needed for pipeline or dataset embeddings
        if self.pipeline_emb_source == "pykeen-lp" or self.dataset_emb_source == "pykeen-lp":
            if not self.pykeen_lp_model_path:
                raise ValueError(
                    "pykeen_lp_model_path must be provided when using 'pykeen-lp' for embeddings"
                )
            
            if self.verbose:
                print(f"Loading PyKEEN LP model from {self.pykeen_lp_model_path}...")
            
            self.pykeen_lp_model = load_pykeen_lp_model(self.pykeen_lp_model_path)
            self.pykeen_lp_dataset = ExeKGPykeenDataset(
                use_mlseakg=self.use_mlseakg, use_mkga=self.use_mkga
            )
        
        # Load RDF2Vec model if needed for pipeline or dataset embeddings
        if self.pipeline_emb_source == "rdf2vec" or self.dataset_emb_source == "rdf2vec":
            pipeline_embeddings_pickle_path = pathlib.Path(
                str(PIPELINE_EMBEDDINGS_PICKLE_PATH).format(
                    walk_distance=self.walk_distance,
                    num_walks=self.num_walks,
                    use_mlseakg=self.use_mlseakg,
                    use_mkga=self.use_mkga,
                )
            )
            dataset_embeddings_pickle_path = pathlib.Path(
                str(DATASET_EMBEDDINGS_PICKLE_PATH).format(
                    walk_distance=self.walk_distance,
                    num_walks=self.num_walks,
                    use_mlseakg=self.use_mlseakg,
                    use_mkga=self.use_mkga,
                )
            )

            # Only load if embeddings don't exist or if explicitly requested
            if (
                (self.pipeline_emb_source == "rdf2vec" and not pipeline_embeddings_pickle_path.exists())
                or (self.dataset_emb_source == "rdf2vec" and not dataset_embeddings_pickle_path.exists())
            ):
                if self.verbose:
                    print("Loading RDF2Vec model and KG...")
                self.rdf2vec_kg = RDF2VecKG(
                    use_mlseakg=self.use_mlseakg, use_mkga=self.use_mkga
                )
                self.rdf2vec_model = load_rdf2vec_model(
                    walk_distance=self.walk_distance,
                    num_walks=self.num_walks,
                    walk_strategy="random",
                    use_mlseakg=self.use_mlseakg,
                    use_mkga=self.use_mkga,
                )

    def prepare_embeddings(self):
        """Prepare pipeline and dataset embeddings."""
        self._get_pipeline_embeddings()

        if self.dataset_emb_source == "rdf2vec":
            self._get_rdf2vec_dataset_embeddings()
        elif self.dataset_emb_source == "pykeen-lp":
            self._get_pykeen_lp_dataset_embeddings()
        elif self.dataset_emb_source in METAFEATURE_PATHS:
            self._get_metafeature_embeddings()

        # Fuse embeddings
        self.data_df["embedding"] = self.data_df[
            ["pipeline_embedding", "dataset_embedding"]
        ].apply(
            lambda x: (
                self._fuse_embeddings(
                    x["pipeline_embedding"], x["dataset_embedding"], self.emb_aggr_type
                )
                if x["pipeline_embedding"] is not None
                and x["dataset_embedding"] is not None
                else None
            ),
            axis=1,
        )

    def _get_pipeline_embeddings(self):
        """Get or compute pipeline embeddings."""
        if self.pipeline_emb_source == "pykeen-lp":
            # Use PyKEEN LP specific cache path
            model_name = pathlib.Path(self.pykeen_lp_model_path).name.split('_mlseakg')[0]
            
            pipeline_embeddings_pickle_path = pathlib.Path(
                str(PYKEEN_LP_PIPELINE_EMBEDDINGS_PICKLE_PATH).format(
                    model_name=model_name,
                    use_mlseakg=self.use_mlseakg,
                    use_mkga=self.use_mkga,
                )
            )
        else:
            pipeline_embeddings_pickle_path = pathlib.Path(
                str(PIPELINE_EMBEDDINGS_PICKLE_PATH).format(
                    walk_distance=self.walk_distance,
                    num_walks=self.num_walks,
                    use_mlseakg=self.use_mlseakg,
                    use_mkga=self.use_mkga,
                )
            )

        if self.verbose:
            print("Getting pipeline embeddings...")

        if pipeline_embeddings_pickle_path.exists():
            with open(pipeline_embeddings_pickle_path, "rb") as f:
                pipeline_embeddings_cache = pickle.load(f)
                if self.verbose:
                    print(
                        f"Loaded {len(pipeline_embeddings_cache)} embeddings from cache."
                    )
        else:
            pipeline_embeddings_cache = self._compute_pipeline_embeddings()
            with open(pipeline_embeddings_pickle_path, "wb") as f:
                pickle.dump(pipeline_embeddings_cache, f)

        self.data_df["pipeline_embedding"] = self.data_df["run_id"].map(
            pipeline_embeddings_cache
        )

        # Calculate statistics
        total_unique_run_ids = self.data_df["run_id"].nunique()
        calculated_embeddings_count = len(
            [x for x in pipeline_embeddings_cache.values() if x is not None]
        )

        if self.verbose:
            print(
                f"Embeddings found for {calculated_embeddings_count} unique run IDs "
                f"out of {total_unique_run_ids} total unique run IDs."
            )

    def _compute_pipeline_embeddings(self) -> Dict[int, Optional[np.ndarray]]:
        """Compute pipeline embeddings using parallel processing."""
        pipeline_embeddings_cache = {}

        if self.pipeline_emb_source == "pykeen-lp":
            # Use PyKEEN LP embeddings
            def compute_pipeline_embedding(run_id, task_id):
                pipeline_name = self._get_pipeline_name_from_run_id_and_task_id(
                    run_id, task_id, EXEKGS_RAW_DIR
                )
                embedding = self._get_pykeen_lp_pipeline_emb(pipeline_name) if pipeline_name else None
                return run_id, embedding

            run_ids = self.data_df["run_id"].unique().tolist()

            with tqdm(total=len(run_ids), desc="Processing Pipeline Embeddings (PyKEEN LP)") as pbar:
                for run_id in run_ids:
                    task_id = self.run_to_task_dict[str(run_id)]
                    run_id, embedding = compute_pipeline_embedding(run_id, task_id)
                    pipeline_embeddings_cache[run_id] = embedding
                    pbar.update(1)
        else:
            # Use RDF2Vec embeddings
            def compute_pipeline_embedding(run_id, task_id, rdf2vec_kg, rdf2vec_model):
                pipeline_name = self._get_pipeline_name_from_run_id_and_task_id(
                    run_id, task_id, EXEKGS_RAW_DIR
                )
                embedding = self._get_pipeline_emb(pipeline_name) if pipeline_name else None
                return run_id, embedding

            run_ids = self.data_df["run_id"].unique().tolist()

            with tqdm(total=len(run_ids), desc="Processing Pipeline Embeddings") as pbar:
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(
                            compute_pipeline_embedding,
                            run_id,
                            self.run_to_task_dict[str(run_id)],
                            self.rdf2vec_kg,
                            self.rdf2vec_model,
                        )
                        for run_id in run_ids
                    ]

                    for future in as_completed(futures):
                        run_id, embedding = future.result()
                        pipeline_embeddings_cache[run_id] = embedding
                        pbar.update(1)

        return pipeline_embeddings_cache

    def _get_rdf2vec_dataset_embeddings(self):
        """Get or compute RDF2Vec dataset embeddings."""
        dataset_embeddings_pickle_path = pathlib.Path(
            str(DATASET_EMBEDDINGS_PICKLE_PATH).format(
                walk_distance=self.walk_distance,
                num_walks=self.num_walks,
                use_mlseakg=self.use_mlseakg,
                use_mkga=self.use_mkga,
            )
        )

        if self.verbose:
            print("Getting rdf2vec dataset embeddings...")

        if dataset_embeddings_pickle_path.exists():
            with open(dataset_embeddings_pickle_path, "rb") as f:
                dataset_embeddings_cache = pickle.load(f)
                if self.verbose:
                    print(
                        f"Loaded {len(dataset_embeddings_cache)} embeddings from cache."
                    )
        else:
            dataset_embeddings_cache = {}
            dataset_ids = self.data_df["dataset_id"].unique().tolist()
            for dataset_id in dataset_ids:
                dataset_embeddings_cache[dataset_id] = self._get_dataset_embedding(
                    dataset_id
                )
            with open(dataset_embeddings_pickle_path, "wb") as f:
                pickle.dump(dataset_embeddings_cache, f)

        self.data_df["dataset_embedding"] = self.data_df["dataset_id"].map(
            dataset_embeddings_cache
        )

    def _get_metafeature_embeddings(self):
        """Get metafeature-based dataset embeddings."""
        if self.verbose:
            print(f"Getting {self.dataset_emb_source} dataset embeddings...")
        with open(METAFEATURE_PATHS[self.dataset_emb_source], "rb") as f:
            dataset_embeddings = pickle.load(f)
        dataset_embeddings = {k: np.array(v) for k, v in dataset_embeddings.items()}
        self.data_df["dataset_embedding"] = self.data_df["dataset_id"].map(
            dataset_embeddings
        )

    def _get_pykeen_lp_dataset_embeddings(self):
        """Get or compute PyKEEN LP dataset embeddings."""
        # Extract model name from path for cache naming
        model_name = pathlib.Path(self.pykeen_lp_model_path).name.split('_mlseakg')[0]
        
        dataset_embeddings_pickle_path = pathlib.Path(
            str(PYKEEN_LP_DATASET_EMBEDDINGS_PICKLE_PATH).format(
                model_name=model_name,
                use_mlseakg=self.use_mlseakg,
                use_mkga=self.use_mkga,
            )
        )

        if self.verbose:
            print("Getting PyKEEN LP dataset embeddings...")

        if dataset_embeddings_pickle_path.exists():
            with open(dataset_embeddings_pickle_path, "rb") as f:
                dataset_embeddings_cache = pickle.load(f)
                if self.verbose:
                    print(
                        f"Loaded {len(dataset_embeddings_cache)} embeddings from cache."
                    )
        else:
            dataset_embeddings_cache = {}
            dataset_ids = self.data_df["dataset_id"].unique().tolist()
            for dataset_id in dataset_ids:
                dataset_info = self.openml_datasets_df[
                    self.openml_datasets_df["dataset_id"] == dataset_id
                ].iloc[0]
                feature_entities = dataset_info["feature_data_entities"].split(", ")
                data_entities = feature_entities + [dataset_info["label_data_entity"]]

                data_entity_embeddings, _ = get_data_entity_emb_pykeen_lp(
                    data_entities, self.pykeen_lp_dataset, self.pykeen_lp_model, 0
                )

                if data_entity_embeddings:
                    dataset_embeddings_cache[dataset_id] = np.mean(
                        data_entity_embeddings, axis=0
                    )
                else:
                    dataset_embeddings_cache[dataset_id] = None
                    if self.verbose:
                        print(f"Failed dataset: {dataset_id}")
            
            with open(dataset_embeddings_pickle_path, "wb") as f:
                pickle.dump(dataset_embeddings_cache, f)

        self.data_df["dataset_embedding"] = self.data_df["dataset_id"].map(
            dataset_embeddings_cache
        )

    def prepare_training_data(self) -> Tuple:
        """Prepare the final training data."""
        # Filter out rows with null embeddings
        data_df_emb_not_null = self.data_df[self.data_df["embedding"].notnull()].copy()
        unique_dataset_ids = data_df_emb_not_null["dataset_id"].unique()

        if len(data_df_emb_not_null) < len(self.data_df):
            if self.verbose:
                print(
                    f"Filtered out {len(self.data_df) - len(data_df_emb_not_null)}/{len(self.data_df)} rows with null embeddings."
                )
                if len(unique_dataset_ids) < len(self.data_df["dataset_id"].unique()):
                    filtered_ids = set(self.data_df["dataset_id"].unique()) - set(
                        unique_dataset_ids
                    )
                    print(
                        f"Filtered out dataset IDs with null embeddings: {filtered_ids}"
                    )

        data_df_to_split = data_df_emb_not_null.copy()

        # Determine target column based on model type
        if self.model in ["RF", "SVC", "LR"]:  # Classification models
            if self.verbose:
                print("Using classification model. Binning the target variable...")
            data_df_to_split = self._bin_target_variable(data_df_to_split, self.target)
            target_col = f"{self.target}_binned"
        elif self.model in ["RFReg", "SVR", "LRReg"]:  # Regression models
            if self.verbose:
                print("Using regression model. Using the original target variable...")
            target_col = self.target
        else:
            raise ValueError(f"Unknown model type: {self.model}")

        # Choose split mode
        if self.split_mode == "dataset":
            group_col, results_dir_template = "dataset_id", RESULTS_DIR_DATASET
        elif self.split_mode == "pipeline":
            group_col, results_dir_template = "run_id", RESULTS_DIR_PIPELINE
        else:
            raise ValueError(f"Unknown split_mode: {self.split_mode}")

        # Split data
        (
            X_train,
            X_test,
            y_train,
            y_test,
            groups_train,
            groups_test,
            train_mask,
            test_mask,
        ) = self._train_test_split_no_overlap(
            data_df_to_split, group_col, target_col, "embedding", 0.7, 0.3, 42
        )

        # Filter by minimum training samples per run_id
        # For pipeline split mode: train and test have disjoint run_ids, so filter independently
        # For dataset split mode: same run_ids appear in both, so filter by training run_ids
        if self.split_mode == "pipeline":
            # Filter train set
            train_run_ids = data_df_to_split.loc[train_mask, "run_id"]
            train_run_id_counts = train_run_ids.value_counts()
            eligible_train_run_ids = set(
                train_run_id_counts[train_run_id_counts >= self.min_train_samples_per_run_id].index
            )
            
            # Filter test set (apply same threshold but to test run_ids)
            test_run_ids = data_df_to_split.loc[test_mask, "run_id"]
            test_run_id_counts = test_run_ids.value_counts()
            eligible_test_run_ids = set(
                test_run_id_counts[test_run_id_counts >= self.min_train_samples_per_run_id].index
            )
            
            if self.verbose:
                print(
                    f"Eligible train run IDs (>= {self.min_train_samples_per_run_id} samples): {len(eligible_train_run_ids)}"
                )
                print(
                    f"Eligible test run IDs (>= {self.min_train_samples_per_run_id} samples): {len(eligible_test_run_ids)}"
                )
            
            train_mask_filtered = train_mask & data_df_to_split["run_id"].isin(eligible_train_run_ids)
            test_mask_filtered = test_mask & data_df_to_split["run_id"].isin(eligible_test_run_ids)
        else:
            # Dataset split mode: filter by training run_ids (same run_ids appear in both sets)
            train_run_ids = data_df_to_split.loc[train_mask, "run_id"]
            run_id_counts = train_run_ids.value_counts()
            eligible_run_ids = set(
                run_id_counts[run_id_counts >= self.min_train_samples_per_run_id].index
            )

            if self.verbose:
                print(
                    f"Eligible run IDs (with at least {self.min_train_samples_per_run_id} training samples): {len(eligible_run_ids)}"
                )

            train_mask_filtered = train_mask & data_df_to_split["run_id"].isin(eligible_run_ids)
            test_mask_filtered = test_mask & data_df_to_split["run_id"].isin(eligible_run_ids)

        if self.verbose:
            print(
                f"After filtering: Train samples = {train_mask_filtered.sum()}, Test samples = {test_mask_filtered.sum()}"
            )

        X_train = np.stack(
            data_df_to_split.loc[train_mask_filtered, "embedding"].values
        )
        X_test = np.stack(data_df_to_split.loc[test_mask_filtered, "embedding"].values)
        y_train = data_df_to_split.loc[train_mask_filtered, target_col].reset_index(
            drop=True
        )
        y_test = data_df_to_split.loc[test_mask_filtered, target_col].reset_index(
            drop=True
        )

        groups_train = data_df_to_split.loc[train_mask_filtered, group_col].reset_index(
            drop=True
        )
        groups_test = data_df_to_split.loc[test_mask_filtered, group_col].reset_index(
            drop=True
        )

        # Reindex groups to consecutive integers
        all_groups = pd.concat([groups_train, groups_test]).unique()
        group_to_idx = {gid: i for i, gid in enumerate(all_groups)}
        groups_train, groups_test = groups_train.map(group_to_idx), groups_test.map(
            group_to_idx
        )

        orig_y_train = data_df_emb_not_null.loc[
            train_mask_filtered, self.target
        ].reset_index(drop=True)
        orig_y_test = data_df_emb_not_null.loc[
            test_mask_filtered, self.target
        ].reset_index(drop=True)

        if self.verbose:
            print(
                f"After filtering: Train samples = {len(X_train)}, Test samples = {len(X_test)}"
            )

        # Scale the data
        scaler = StandardScaler()
        X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(
            X_test
        )

        return (
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test,
            groups_train,
            groups_test,
            orig_y_train,
            orig_y_test,
            results_dir_template,
            target_col,
        )

    def run_prediction(self):
        """Run the complete pipeline performance prediction workflow."""
        if self.verbose:
            print("Starting pipeline performance prediction...")

        # Load data and models
        self.load_data()
        self.load_models()
        self.prepare_embeddings()

        # Prepare training data
        (
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test,
            groups_train,
            groups_test,
            orig_y_train,
            orig_y_test,
            results_dir_template,
            target_col,
        ) = self.prepare_training_data()

        # Define model and parameter grid
        model_config = self._get_model_config()

        # Set up results directory
        # Extract PyKEEN LP model name if being used
        pykeen_lp_model_part = ""
        if self.pipeline_emb_source == "pykeen-lp" or self.dataset_emb_source == "pykeen-lp":
            if self.pykeen_lp_model_path:
                model_name = pathlib.Path(self.pykeen_lp_model_path).name.split('_mlseakg')[0]
                pykeen_lp_model_part = f"_pykeen_lp_model_{model_name}"
        
        results_dir = pathlib.Path(
            str(results_dir_template).format(
                target=self.target,
                model=model_config["name"],
                emb_aggr_type=self.emb_aggr_type,
                dataset_emb_source=self.dataset_emb_source,
                pipeline_emb_source=self.pipeline_emb_source,
                pykeen_lp_model_part=pykeen_lp_model_part,
                use_mlseakg=self.use_mlseakg,
                use_mkga=self.use_mkga,
                min_train_samples_per_run_id=self.min_train_samples_per_run_id,
            )
        )
        results_dir.mkdir(parents=True, exist_ok=True)

        # Print target statistics, train model, evaluate
        self._print_target_statistics(y_train, y_test)
        best_model, grid_search = self._train_model(
            model_config, X_train_scaled, y_train, groups_train, results_dir
        )
        test_results = self._evaluate_model(
            best_model, X_test_scaled, y_test, results_dir
        )

        # Run baselines if needed
        if self.split_mode == "pipeline":
            baseline_results = self._evaluate_baselines(
                X_train_scaled,
                orig_y_train,
                X_test_scaled,
                orig_y_test,
                y_train,
                y_test,
                results_dir,
            )

        if self.verbose:
            print("Pipeline performance prediction completed!")

    def _get_model_config(self) -> Dict[str, Any]:
        """Get model configuration based on model type."""

        if self.model not in PERF_PRED_MODEL_CONFIGS:
            raise ValueError(f"Unknown model type: {self.model}")
        return PERF_PRED_MODEL_CONFIGS[self.model]

    def _print_target_statistics(self, y_train: pd.Series, y_test: pd.Series):
        """Print statistics about target variables."""
        if not self.verbose:
            return

        if self.model in ["RF", "SVC", "LR"]:
            print(
                "Train target variable value counts:", pd.Series(y_train).value_counts()
            )
            print(
                "Test target variable value counts:", pd.Series(y_test).value_counts()
            )
        elif self.model in ["RFReg", "SVR", "LRReg"]:
            for name, y in [("Train", y_train), ("Test", y_test)]:
                print(f"{name} target variable range: {np.min(y)} - {np.max(y)}")
                print(f"{name} target variable mean: {np.mean(y)}")
                print(f"{name} target variable std: {np.std(y)}")
                print(f"{name} target variable variance: {np.var(y)}")

    def _train_model(
        self,
        model_config: Dict[str, Any],
        X_train: np.ndarray,
        y_train: pd.Series,
        groups_train: pd.Series,
        results_dir: pathlib.Path,
    ) -> Tuple:
        """Train the model using GridSearchCV."""
        if self.verbose:
            print("Starting GridSearchCV...")

        grid_search = GridSearchCV(
            estimator=model_config["estimator"],
            param_grid=model_config["param_grid"],
            cv=GroupKFold(n_splits=N_FOLDS),
            scoring=model_config["scorer"],
            n_jobs=-1,
            verbose=2 if self.verbose else 0,
        )

        grid_search.fit(X_train, y_train, groups=groups_train)
        best_model = grid_search.best_estimator_

        if self.verbose:
            print("Best Parameters:", grid_search.best_params_)
            print("Best Validation Score:", grid_search.best_score_)

        # Save results
        pd.DataFrame(grid_search.cv_results_).to_csv(
            results_dir / "hpo_results.csv", index=False
        )
        with open(results_dir / "best_model.pkl", "wb") as f:
            pickle.dump(best_model, f)

        if self.verbose:
            print(f"GridSearchCV results saved to {results_dir / 'hpo_results.csv'}")
            print(f"Best model saved to {results_dir / 'best_model.pkl'}")

        return best_model, grid_search

    def _evaluate_model(
        self,
        best_model,
        X_test: np.ndarray,
        y_test: pd.Series,
        results_dir: pathlib.Path,
    ) -> Dict[str, Any]:
        """Evaluate the trained model on test data."""
        if self.verbose:
            print("Evaluating on Test Data...")

        y_pred_test = best_model.predict(X_test)

        if self.model in ["RF", "SVC", "LR"]:  # Classification models
            if self.verbose:
                print(
                    "\nClassification Report:",
                    classification_report(y_test, y_pred_test),
                )
            test_accuracy, test_f1 = accuracy_score(y_test, y_pred_test), f1_score(
                y_test, y_pred_test, average="weighted"
            )
            if self.verbose:
                print(f"Test Accuracy: {test_accuracy:.4f}")
                print(f"Test F1: {test_f1:.4f}")
            results = {
                "test_metrics": {"accuracy": test_accuracy, "weighted_f1": test_f1}
            }
        elif self.model in ["RFReg", "SVR", "LRReg"]:  # Regression models
            test_mse, test_r2 = mean_squared_error(y_test, y_pred_test), r2_score(
                y_test, y_pred_test
            )
            if self.verbose:
                print(f"\nResults: MSE = {test_mse:.4f}, R2 = {test_r2:.4f}")
            results = {"test_metrics": {"mse": test_mse, "r2": test_r2}}

        # Save results
        with open(results_dir / "best_model_results.json", "w") as f:
            json.dump(results, f, indent=4)
        return results

    def _evaluate_baselines(
        self,
        X_train: np.ndarray,
        orig_y_train: pd.Series,
        X_test: np.ndarray,
        orig_y_test: pd.Series,
        y_train: pd.Series,
        y_test: pd.Series,
        results_dir: pathlib.Path,
    ) -> Dict[str, Any]:
        """Evaluate baseline methods."""
        if self.verbose:
            print("Evaluating baselines for unseen pipeline prediction...")

        y_pred_closest = self._closest_embedding_baseline(X_train, orig_y_train, X_test)
        y_pred_avg = self._average_baseline(orig_y_train, X_test)
        baseline_results = {}

        if self.model in ["RF", "SVC", "LR"]:
            num_bins = y_train.nunique()
            bin_edges = pd.qcut(
                orig_y_train, q=num_bins, retbins=True, duplicates="drop"
            )[1]
            bin_labels = sorted(
                y_train.unique(), key=lambda x: int(str(x).split("_")[1])
            )

            y_pred_closest_binned = pd.cut(
                y_pred_closest, bins=bin_edges, labels=bin_labels, include_lowest=True
            )
            y_pred_avg_binned = pd.cut(
                y_pred_avg, bins=bin_edges, labels=bin_labels, include_lowest=True
            )

            if self.verbose:
                print(
                    "\nClosest Embedding Baseline (Classification):",
                    classification_report(y_test, y_pred_closest_binned),
                )
                print(
                    "\nAverage Baseline (Classification):",
                    classification_report(y_test, y_pred_avg_binned),
                )

            baseline_results = {
                "closest_embedding": {
                    "accuracy": float(accuracy_score(y_test, y_pred_closest_binned)),
                    "weighted_f1": float(
                        f1_score(y_test, y_pred_closest_binned, average="weighted")
                    ),
                },
                "average": {
                    "accuracy": float(accuracy_score(y_test, y_pred_avg_binned)),
                    "weighted_f1": float(
                        f1_score(y_test, y_pred_avg_binned, average="weighted")
                    ),
                },
            }
        else:
            closest_mse, closest_r2 = mean_squared_error(
                orig_y_test, y_pred_closest
            ), r2_score(orig_y_test, y_pred_closest)
            avg_mse, avg_r2 = mean_squared_error(orig_y_test, y_pred_avg), r2_score(
                orig_y_test, y_pred_avg
            )

            if self.verbose:
                print(
                    f"\nClosest Embedding Baseline (Regression): MSE: {closest_mse:.4f}, R2: {closest_r2:.4f}"
                )
                print(
                    f"\nAverage Baseline (Regression): MSE: {avg_mse:.4f}, R2: {avg_r2:.4f}"
                )

            baseline_results = {
                "closest_embedding": {
                    "mse": float(closest_mse),
                    "r2": float(closest_r2),
                },
                "average": {"mse": float(avg_mse), "r2": float(avg_r2)},
            }

        # Save baseline results
        with open(results_dir / "baseline_results.json", "w") as f:
            json.dump(baseline_results, f, indent=4)

        if self.verbose:
            print(f"Baseline results saved to {results_dir / 'baseline_results.json'}")
        return baseline_results
