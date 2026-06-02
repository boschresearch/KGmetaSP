# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0


import pathlib
import site


from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import make_scorer, mean_squared_error, accuracy_score
from sklearn.svm import SVC, SVR

HERE = pathlib.Path(__file__).resolve().parent

DATA_DIR = HERE / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
KGE_MODELS_DIR = DATA_DIR / "kge_models"

PYKEEN_MODEL_PATH = (
    KGE_MODELS_DIR
    / "pykeen"
    / "exekgs_results_with_params"
    / "best_mrr"
    / "pykeen_lr_0.0001_transerealite_exekgdatasetonerunpertask_results"
    / "trained_model.pkl"
)

# PyKEEN Link Prediction paths
PYKEEN_LP_MODELS_DIR = KGE_MODELS_DIR / "pykeen_lp"
PYKEEN_LP_MODEL_PATH = (
    PYKEEN_LP_MODELS_DIR
    / "{model_type}_mlseakg_{use_mlseakg}_mkga_{use_mkga}_invtri_{inv_tri}"
)

RDF2VEC_MODEL_DIR = KGE_MODELS_DIR / "rdf2vec" / "d_{d}_w_{w}_ws_{ws}_{kg}"
RDF2VEC_MODEL_PATH = RDF2VEC_MODEL_DIR / "rdf2vec_model.model"

EXEKGS_NT_PATH = INPUT_DIR / "datasets" / "rdf2vec" / "exekgs.nt"
MLSEAKG_NT_PATH = INPUT_DIR / "datasets" / "rdf2vec" / "openml_datasets_1.nt"
MLSEAKG_FILTERED_NT_PATH = INPUT_DIR / "datasets" / "rdf2vec" / "mlseakg_filtered.nt"
EXEKGS_W_MLSEAKG_NT_PATH = INPUT_DIR / "datasets" / "rdf2vec" / "exekgs_with_mlseakg.nt"

EXEKGS_MKGA_NT_PATH = INPUT_DIR / "datasets" / "rdf2vec" / "exekgs_mkga.nt"
EXEKGS_W_MLSEAKG_MKGA_NT_PATH = (
    INPUT_DIR / "datasets" / "rdf2vec" / "exekgs_with_mlseakg_mkga.nt"
)

SITE_PACKAGES_DIR = pathlib.Path(site.getsitepackages()[0])
KGBENCH_DATASETS_DIR = SITE_PACKAGES_DIR / "kgbench" / "datasets"

# kgbench datasets
EXEKGS_TGZ_PATH = KGBENCH_DATASETS_DIR / "exekgs.tgz"
EXEKGS_W_MLSEAKG_TGZ_PATH = KGBENCH_DATASETS_DIR / "exekgs_with_mlseakg.tgz"

DATASETS_CSV_PATH = INPUT_DIR / "logs" / "tasks_log.csv"
EXEKGS_RAW_DIR = INPUT_DIR / "datasets" / "raw" / "exekgs"

SIMILARITIES_CSV_PATH = OUTPUT_DIR / "similarities.csv"

FIGURES_DIR = OUTPUT_DIR / "figures"

MLSEAKG_GRAPHDB_ENDPOINT = "https://kg.cs.kuleuven.be/graphdb1/sparql"

MKGA_CONFIG_RELPATH = pathlib.Path(
    "..", "..", "MKGA-exekgs-extension", "config", "single.yaml"
)

PERF_PRED_MODEL_CONFIGS = {
    "RF": {
        "estimator": RandomForestClassifier(random_state=42),
        "param_grid": {
            "n_estimators": [50, 100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5, 10],
        },
        "scorer": make_scorer(accuracy_score),
        "name": "RandomForestClassifier",
    },
    "SVC": {
        "estimator": SVC(),
        "param_grid": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto"],
        },
        "scorer": make_scorer(accuracy_score),
        "name": "SVC",
    },
    "LR": {
        "estimator": LogisticRegression(),
        "param_grid": {
            "penalty": ["l2"],
            "C": [0.1, 1, 10],
            "solver": ["lbfgs"],
        },
        "scorer": make_scorer(accuracy_score),
        "name": "LogisticRegression",
    },
    "RFReg": {
        "estimator": RandomForestRegressor(random_state=42),
        "param_grid": {
            "n_estimators": [20, 50, 100, 200],
            "max_depth": [10, 20],
            "min_samples_split": [2, 5, 10],
        },
        "scorer": make_scorer(mean_squared_error, greater_is_better=False),
        "name": "RandomForestRegressor",
    },
    "SVR": {
        "estimator": SVR(),
        "param_grid": {
            "C": [0.1, 1, 10, 100],
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto"],
        },
        "scorer": make_scorer(mean_squared_error, greater_is_better=False),
        "name": "SVR",
    },
    "LRReg": {
        "estimator": LinearRegression(),
        "param_grid": {
            "fit_intercept": [True, False],
            "normalize": [True, False],
        },
        "scorer": make_scorer(mean_squared_error, greater_is_better=False),
        "name": "LinearRegression",
    },
}

# PyKEEN Link Prediction HPO Configuration
PYKEEN_LP_HPO_CONFIG = {
    "pipeline": {
        "stopper": "early",
        "stopper_kwargs": {
            "metric": "inverse_harmonic_mean_rank",
            "frequency": 5,
            "patience": 10,
            "relative_delta": 0.002,
        },
        "training_kwargs": {
            "num_workers": 4,
            "drop_last": False,
        },
        "training_kwargs_ranges": {
            "batch_size": {"type": "int", "low": 512, "high": 5120, "step": 512}
        },
        "optimizer": "adam",
        "optimizer_kwargs_ranges": {
            "lr": {"type": "float", "low": 0.0001, "high": 0.01, "log": True}
        },
        "model_to_model_kwargs": {
            "DistMultLiteralGated": {"embedding_dim": 200},
            "DistMultReaLitE": {"embedding_dim": 200, "aggregation": "mode"},
            "DistMultReaLitEGated": {"embedding_dim": 200, "aggregation": "mode"},
            "TransELiteralE": {"embedding_dim": 200},
            "TransEReaLitE": {"embedding_dim": 200, "aggregation": "mode"},
            "TransEReaLitEGated": {"embedding_dim": 200, "aggregation": "mode"},
            "ComplExReaLitE": {"embedding_dim": 200, "aggregation": "mode"},
            "ComplExReaLitEGated": {"embedding_dim": 200, "aggregation": "mode"},
        },
        "model_to_model_kwargs_ranges": {
            "DistMultLiteralGated": {
                "input_dropout": {"type": "float", "low": 0.1, "high": 0.5, "step": 0.1}
            },
            "DistMultReaLitE": {
                "input_dropout": {"type": "float", "low": 0.4, "high": 0.5, "step": 0.1}
            },
            "DistMultReaLitEGated": {
                "input_dropout": {"type": "float", "low": 0.1, "high": 0.5, "step": 0.1}
            },
            "TransEReaLitE": {
                "input_dropout": {"type": "float", "low": 0.3, "high": 0.5, "step": 0.1}
            },
            "TransELiteralE": {
                "input_dropout": {"type": "float", "low": 0.1, "high": 0.5, "step": 0.1}
            },
            "TransEReaLitEGated": {
                "input_dropout": {"type": "float", "low": 0.1, "high": 0.5, "step": 0.1}
            },
            "ComplExReaLitE": {
                "input_dropout": {"type": "float", "low": 0.3, "high": 0.5, "step": 0.1}
            },
            "ComplExReaLitEGated": {
                "input_dropout": {"type": "float", "low": 0.1, "high": 0.5, "step": 0.1}
            },
        },
    }
}
