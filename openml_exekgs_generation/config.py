# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

### OpenML ExeKG Generation Configuration

from pathlib import Path

# ========== OpenML Crawling Configuration ==========

METRIC_FOR_SORTING_PIPELINES_DICT = {
    "Supervised Classification": "f_measure",
    "Supervised Regression": "root_mean_squared_error",
}
PERFORMANCE_CALCULATION_KWARGS_DICT = {
    "f_measure": {"hasParamAverage": "weighted"},
    "root_mean_squared_error": {},
}
OPENML_DATASET_KWARGS = {
    "download_data": False,
    "download_features_meta_data": True,
    "download_qualities": False,
}

RUNS_LOG_FILENAME = "runs_log.csv"
TASKS_LOG_FILENAME = "tasks_log.csv"

HERE = Path(__file__).parent

FETCHED_DATA_DIR = HERE / "fetched_data"
FLOWS_PATH = FETCHED_DATA_DIR / "flows.csv"
TASKS_PATH = FETCHED_DATA_DIR / "tasks.csv"
DATASETS_PATH = FETCHED_DATA_DIR / "datasets.csv"

OUTPUT_DIR = HERE / "output"
LOG_DIR = OUTPUT_DIR / "logs"
EXEKGS_OUTPUT_DIR = OUTPUT_DIR / "exekgs"

# ========== OpenML to ExeKG Mappings ==========

OPENML_EST_PROCEDURE_TYPE_TO_SKLEARN = {
    "holdout": "ShuffleSplit",
    # "holdout stratified": "StratifiedShuffleSplit",
    "crossvalidation": "KFold",
    "leaveoneout": "LeaveOneOut",
    # "crossvalidation stratified": "StratifiedKFold",
    # "repeated_holdout": "RepeatedKFold",
    # "repeated_stratified_holdout": "RepeatedStratifiedKFold",
    # "subgroups": "GroupKFold",
    # "user_defined": "UserDefinedKFold",
}
OPENML_EST_PROCEDURE_PARAMS_TO_SKLEARN = {
    "percentage": "test_size",
    # "stratified_sampling": "s",
    "number_folds": "n_splits",
    "number_repeats": "n_repeats",
}
OPENML_DATATYPE_TO_EXEKG_SEMANTICS = {
    "nominal": "Categorical",
    "numeric": "Numerical",
    "string": "Text",
    "date": "Date",
}
OPENML_METRIC_NAME_TO_EXEKG = {
    "predictive_accuracy": "AccuracyScore",
    "f_measure": "F1Score",
    "area_under_roc_curve": "RocAucScore",
    "root_mean_squared_error": "RootMeanSquaredError",
}

OPENML_TO_SKLEARN_CLASSES = {
    "Imputer": "SimpleImputer",
    # "openmlstudy14.preprocessing.ConditionalImputer": "sklearn.impute.SimpleImputer",  # custom OpenML imputer replaced with sklearn imputer
    "ConditionalImputer": "SimpleImputer",  # custom OpenML imputer replaced with sklearn imputer
    "ConditionalImputer2": "SimpleImputer",  # custom OpenML imputer replaced with sklearn imputer
}

OPENML_PARAMS_TO_NEW_SKLEARN = {
    "min_impurity_split": "min_impurity_decrease",
    "presort": None,
}

OPENML_PARAM_VALUES_TO_SKLEARN = {
    "null": None,
    "true": True,
    "false": False,
}
