# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import argparse
import glob
import json
import os
import pathlib
import traceback
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

from kge_experiments.config import DATA_DIR
from kge_experiments.scripts.utils import (
    load_json_safe,
    safe_df,
    filter_df_by_mlseakg,
    filter_df,
    shorten_model_name,
    extract_bool_flag,
    filter_target,
    extract_emb_source,
    extract_target,
    prettify_emb_source,
    BASELINES_END_INDEX,
)

# Define paths
RETRIEVAL_RESULTS_DIR = DATA_DIR / "retrieval_results"
PERF_PRED_RESULTS_DIR = DATA_DIR / "perf_pred_results"
FIGURES_DIR = DATA_DIR / "ablation_results"

# Files for dataset similarity results
RETRIEVAL_RANKING_RESULTS_PATH = (
    RETRIEVAL_RESULTS_DIR / "perf_results_cosine_sim_excl_perf.csv"
)

# Pipeline performance prediction configuration
MIN_TRAIN_SAMPLES = int(os.environ.get("MIN_TRAIN_SAMPLES", 50))
SPLIT_MODE = os.environ.get("SPLIT_MODE", "dataset")  # or "pipeline"

# Remote results directories for pipeline performance prediction
BASE_REMOTE_PATH = pathlib.Path(
    "/fs/scratch/rb_bd_dlp_rng-dl01_cr_AIQ_employees/students/klr2rng/perf_pred_results"
)
REMOTE_RESULTS_DIR = (
    BASE_REMOTE_PATH
    / f"results_target_*_split_{SPLIT_MODE}_mintrainsamples_{MIN_TRAIN_SAMPLES}"
)

# Local baseline results
BASELINE_DATASET_SPLIT_RESULTS_PATH = (
    PERF_PRED_RESULTS_DIR / f"analysis_results_min_{MIN_TRAIN_SAMPLES}_samples.json"
)

# Filter targets for pipeline performance prediction
FILTER_TARGETS = ["accuracy", "precision"]


def ensure_figures_dir():
    """Ensure the figures directory exists."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_combined_similarity_latex_table(
    df_with: pd.DataFrame,
    df_without: pd.DataFrame,
    filename: str,
    title: str = "",
    metrics_to_show: Optional[List[str]] = None,
):
    """
    Save a combined LaTeX table showing similarity models with and without MLSeaKG.
    Similar format to pipeline performance prediction tables.

    Args:
        df_with: DataFrame with MLSeaKG model results
        df_without: DataFrame without MLSeaKG model results
        filename: Output filename
        title: Table caption
        metrics_to_show: List of metrics to include in table (if None, show all)
    """
    if df_with.empty and df_without.empty:
        print(f"No data for similarity comparison table: {filename}")
        return

    # Helper to extract model configuration
    def extract_config(model_name):
        """Extract configuration from model name."""
        if not isinstance(model_name, str):
            return "Unknown"
        parts = model_name.split("_")
        if len(parts) < 3:
            return model_name

        # Parse embedding type
        emb_type_map = {"data": "FeatSim", "pip": "PipSim", "comb": "DatSim"}
        emb_type = emb_type_map.get(parts[0], parts[0])

        return emb_type

    # Prepare table rows
    table_rows = []

    for df, mlseakg_label in [(df_with, "With"), (df_without, "Without")]:
        if not df.empty:
            row_data = {
                "Embedding": extract_config(df["model_name"].iloc[0]),
                "MLSeaKG": mlseakg_label,
            }

            # Add metric values
            for col in df.columns:
                if col != "model_name" and (
                    metrics_to_show is None or col in metrics_to_show
                ):
                    row_data[col] = df[col].iloc[0]

            table_rows.append(row_data)

    if not table_rows:
        print(f"No valid rows for: {filename}")
        return

    result_df = pd.DataFrame(table_rows)

    # Determine which metrics to display
    metric_cols = [
        col for col in result_df.columns if col not in ["Embedding", "MLSeaKG"]
    ]
    if not metric_cols:
        print(f"No metrics found for: {filename}")
        return

    # Format metric values and highlight best
    for col in metric_cols:
        if col in result_df.columns:
            values = result_df[col].dropna()
            if len(values) > 0:
                best_val = values.max()  # Higher is better for all similarity metrics
                result_df[col] = result_df[col].apply(
                    lambda v: (
                        f"\\textbf{{{v:.4f}}}"
                        if pd.notna(v) and v == best_val
                        else (f"{v:.4f}" if pd.notna(v) else "---")
                    )
                )

    # Generate LaTeX
    latex_str = result_df.to_latex(
        escape=False,
        index=False,
        caption=title if title else None,
        label=f"tab:{filename.replace('.tex', '')}" if title else None,
        column_format="ll" + "c" * len(metric_cols),
    )

    output_path = FIGURES_DIR / filename
    with open(output_path, "w") as f:
        f.write(latex_str)

    print(f"Similarity LaTeX table saved to {output_path}")


def save_combined_latex_table(
    df_with: pd.DataFrame,
    df_without: pd.DataFrame,
    filename: str,
    title: str = "",
    task_type: str = "classification",
):
    """
    Save a combined LaTeX table showing models with and without MLSeaKG as separate rows,
    organized by target metric columns.
    """
    # Determine metrics based on task type
    metrics, metric_labels = (
        (["accuracy", "weighted_f1"], ["Accuracy", "F1"])
        if task_type == "classification"
        else (["mse", "r2"], ["MSE", "R²"])
    )

    # Helper functions
    def extract_target(model_name):
        """Extract target from model_name."""
        if not isinstance(model_name, str):
            return "unknown"
        parts = model_name.split("_")
        for i, part in enumerate(parts):
            if part == "target" and i + 1 < len(parts):
                return parts[i + 1]
        return "unknown"

    def extract_emb_source(model_name_for_plotting):
        """Extract embedding source from model_name_for_plotting."""
        if not isinstance(model_name_for_plotting, str):
            return str(model_name_for_plotting)
        return model_name_for_plotting.split("-mls_")[0]

    def prettify_emb_source(s):
        """Prettify embedding source names."""
        s = s.replace("metafeatures", "MF").replace("_", " ")
        return s.title() if not s.startswith("MF") else "MF" + s[2:].title()

    def add_rows_from_df(df, mlseakg_label):
        """Extract rows from DataFrame."""
        rows = []
        if not df.empty:
            for _, row in df.iterrows():
                table_row = {
                    "Embedding": prettify_emb_source(
                        extract_emb_source(row["model_name_for_plotting"])
                    ),
                    "MLSeaKG": mlseakg_label,
                    "target": extract_target(row["model_name"]),
                }
                for metric in metrics:
                    if metric in row.index and pd.notna(row[metric]):
                        table_row[metric] = row[metric]
                rows.append(table_row)
        return rows

    # Prepare data rows
    table_rows = add_rows_from_df(df_with, "With") + add_rows_from_df(
        df_without, "Without"
    )

    if not table_rows:
        print(f"No data for {task_type} comparison table.")
        return

    result_df = pd.DataFrame(table_rows)
    # Select best configuration per (target, MLSeaKG) group
    best_metric = "accuracy" if task_type == "classification" else "mse"
    use_max = task_type == "classification"

    best_rows = []
    for (target, mlseakg), group in result_df.groupby(["target", "MLSeaKG"]):
        if best_metric in group.columns and group[best_metric].notna().any():
            best_idx = (
                group[best_metric].idxmax() if use_max else group[best_metric].idxmin()
            )
            best_rows.append(group.loc[best_idx])

    if not best_rows:
        print(f"No best configurations found for: {filename}")
        return

    best_df = pd.DataFrame(best_rows)

    # Pivot data by (Embedding, MLSeaKG) with targets as column groups
    unique_combos = best_df[["Embedding", "MLSeaKG"]].drop_duplicates()
    pivot_data = []

    for _, combo in unique_combos.iterrows():
        mask = (best_df["Embedding"] == combo["Embedding"]) & (
            best_df["MLSeaKG"] == combo["MLSeaKG"]
        )
        rows_for_combo = best_df[mask]

        pivot_row = {"Embedding": combo["Embedding"], "MLSeaKG": combo["MLSeaKG"]}

        # Add metric columns for each target
        for target in result_df["target"].unique():
            target_row = rows_for_combo[rows_for_combo["target"] == target]
            if not target_row.empty:
                target_row = target_row.iloc[0]
                for metric, metric_label in zip(metrics, metric_labels):
                    pivot_row[(target.title(), metric_label)] = target_row.get(
                        metric, None
                    )
            else:
                for metric, metric_label in zip(metrics, metric_labels):
                    pivot_row[(target.title(), metric_label)] = None

        pivot_data.append(pivot_row)

    table_df = pd.DataFrame(pivot_data).sort_values(["Embedding", "MLSeaKG"])

    # Separate regular columns from MultiIndex columns
    regular_cols = ["Embedding", "MLSeaKG"]
    multi_cols = [col for col in table_df.columns if isinstance(col, tuple)]

    # Format metric values and highlight best per target
    for col in multi_cols:
        target_name, metric_label = col
        values = table_df[col].dropna()
        if len(values) > 0:
            best_val = (
                values.min()
                if (task_type == "regression" and metric_label == "MSE")
                else values.max()
            )
            table_df[col] = table_df[col].apply(
                lambda v: (
                    f"\\textbf{{{v:.4f}}}"
                    if pd.notna(v) and v == best_val
                    else (f"{v:.4f}" if pd.notna(v) else "---")
                )
            )

    # Reorder columns and create MultiIndex
    all_columns = regular_cols + sorted(multi_cols)
    table_df = table_df[all_columns]

    new_columns = [("", col) if col in regular_cols else col for col in all_columns]
    table_df.columns = pd.MultiIndex.from_tuples(new_columns)

    # Generate and save LaTeX
    latex_str = table_df.to_latex(
        escape=False,
        index=False,
        multicolumn=True,
        multicolumn_format="c",
        caption=title if title else None,
        label=f"tab:{filename.replace('.tex', '')}" if title else None,
    )

    output_path = FIGURES_DIR / filename
    with open(output_path, "w") as f:
        f.write(latex_str)

    print(f"Performance Prediction LaTeX table saved to {output_path}")


def analyze_dataset_similarity_results(
    excl_metrics_with_k: Optional[List[str]] = None,
    excl_dot_product=True,
    excl_manhattan=True,
    excl_euclidean=True,
    only_without_mkga=True,
    only_with_mlsea=False,
    selection_metric: str = "NDCG@1",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Analyze dataset similarity results and identify best models with/without MLSeaKG.
    Uses single-metric selection strategy (default: NDCG@1) similar to pipeline performance prediction.

    Args:
        excl_metrics_with_k: List of k values to exclude from analysis
        excl_dot_product: Exclude dot product similarity
        excl_manhattan: Exclude manhattan distance
        excl_euclidean: Exclude euclidean distance
        only_without_mkga: Only include models without MKGA preprocessing
        only_with_mlsea: Only include models with MLSeaKG (False for ablation study)
        selection_metric: Metric to use for selecting best model (default: NDCG@1)

    Returns:
        Tuple of (with_mlseakg_best_df, without_mlseakg_best_df, comparison_df)
    """
    print("Analyzing dataset similarity results...")

    # Load retrieval ranking results
    retrieval_df = pd.read_csv(RETRIEVAL_RANKING_RESULTS_PATH, index_col=0)

    # Apply comprehensive filtering
    retrieval_df = filter_df(
        retrieval_df,
        excl_hit_metrics=False,
        excl_metrics_with_k=excl_metrics_with_k,
        excl_dot_product=excl_dot_product,
        excl_manhattan=excl_manhattan,
        excl_euclidean=excl_euclidean,
        only_without_mkga=only_without_mkga,
        only_with_mlsea=only_with_mlsea,
    )

    # Filter to only KGE models (exclude baselines)
    kge_models = retrieval_df.iloc[BASELINES_END_INDEX:, :]

    # Separate models with and without MLSeaKG
    with_mlseakg = kge_models[kge_models.index.str.contains("mlsea")]
    without_mlseakg = kge_models[~kge_models.index.str.contains("mlsea")]

    # Select single best model based on selection_metric (similar to pipeline perf pred)
    if selection_metric not in retrieval_df.columns:
        print(f"Warning: {selection_metric} not found. Using first available metric.")
        selection_metric = retrieval_df.columns[0]

    # Find single best model for each condition
    best_with_idx = with_mlseakg[selection_metric].idxmax()
    best_without_idx = without_mlseakg[selection_metric].idxmax()

    # Get the single best model rows with all metrics
    with_mlseakg_best_df = pd.DataFrame([with_mlseakg.loc[best_with_idx]])
    without_mlseakg_best_df = pd.DataFrame([without_mlseakg.loc[best_without_idx]])

    # Add model_name column for consistency with pipeline format
    with_mlseakg_best_df["model_name"] = best_with_idx
    without_mlseakg_best_df["model_name"] = best_without_idx

    # # Create comparison dataframe
    # comparison_metrics = []
    # for metric in retrieval_df.columns:
    #     best_with = (
    #         with_mlseakg_best_df[metric].iloc[0]
    #         if metric in with_mlseakg_best_df.columns
    #         else 0
    #     )
    #     best_without = (
    #         without_mlseakg_best_df[metric].iloc[0]
    #         if metric in without_mlseakg_best_df.columns
    #         else 0
    #     )
    #     improvement = (
    #         ((best_with - best_without) / best_without) * 100 if best_without > 0 else 0
    #     )

    #     comparison_metrics.append(
    #         {
    #             "Metric": metric,
    #             "Best with MLSeaKG": best_with,
    #             "Best without MLSeaKG": best_without,
    #             "Improvement (%)": improvement,
    #         }
    #     )

    # comparison_df = pd.DataFrame(comparison_metrics)

    return with_mlseakg_best_df, without_mlseakg_best_df


def analyze_pipeline_performance_prediction() -> (
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
):
    """
    Analyze pipeline performance prediction results for models with/without MLSeaKG.

    Returns:
        Tuple of (with_mlseakg_df, without_mlseakg_df, comparison_df)
    """
    print("Analyzing pipeline performance prediction results...")

    classification_results = []
    regression_results = []

    # Process main model results
    json_files = glob.glob(
        f"{REMOTE_RESULTS_DIR}/**/best_model_results.json", recursive=True
    )

    for file_path in json_files:
        data = load_json_safe(file_path)
        if not data:
            continue

        model_name = os.path.basename(os.path.dirname(file_path))
        if not isinstance(model_name, str) or "mlseakg" not in model_name:
            continue

        model_name_for_plotting = shorten_model_name(model_name)
        test_metrics = data.get("test_metrics", {})

        # Classification or regression based on metrics present
        if "accuracy" in test_metrics:
            classification_results.append(
                {
                    "model_name": model_name,
                    "model_name_for_plotting": model_name_for_plotting,
                    "accuracy": test_metrics.get("accuracy"),
                    "weighted_f1": test_metrics.get("weighted_f1"),
                    "params": data.get("best_model_params", {}),
                }
            )
        elif "mse" in test_metrics:
            regression_results.append(
                {
                    "model_name": model_name,
                    "model_name_for_plotting": model_name_for_plotting,
                    "mse": test_metrics.get("mse"),
                    "r2": test_metrics.get("r2"),
                    "params": data.get("best_model_params", {}),
                }
            )

    # Process baselines based on split mode
    if SPLIT_MODE == "pipeline":
        _process_pipeline_baselines(classification_results, regression_results)
    elif SPLIT_MODE == "dataset":
        _process_dataset_baselines(classification_results, regression_results)

    # Convert to DataFrames and filter
    classification_df = filter_target(
        pd.DataFrame(classification_results), FILTER_TARGETS
    )
    regression_df = filter_target(pd.DataFrame(regression_results), FILTER_TARGETS)

    if classification_df.empty and regression_df.empty:
        print("No pipeline performance prediction results found.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Separate by MLSeaKG usage
    def has_mlseakg(model_name_for_plotting):
        if (
            model_name_for_plotting.startswith("closest")
            or model_name_for_plotting.startswith("average")
            or model_name_for_plotting.endswith("-multiple")
        ):
            return None
        return extract_bool_flag(model_name_for_plotting, "mls") == "True"

    # Filter and separate DataFrames
    classification_kge = safe_df(
        classification_df[
            classification_df["model_name_for_plotting"].apply(
                lambda x: has_mlseakg(x) is not None
            )
        ]
    )
    regression_kge = safe_df(
        regression_df[
            regression_df["model_name_for_plotting"].apply(
                lambda x: has_mlseakg(x) is not None
            )
        ]
    )

    classification_with = safe_df(
        classification_kge[
            classification_kge["model_name_for_plotting"].apply(has_mlseakg) == True
        ]
    )
    classification_without = safe_df(
        classification_kge[
            classification_kge["model_name_for_plotting"].apply(has_mlseakg) == False
        ]
    )
    regression_with = safe_df(
        regression_kge[
            regression_kge["model_name_for_plotting"].apply(has_mlseakg) == True
        ]
    )
    regression_without = safe_df(
        regression_kge[
            regression_kge["model_name_for_plotting"].apply(has_mlseakg) == False
        ]
    )

    # Create comparison DataFrame
    comparison_data = _create_comparison_data(
        classification_with, classification_without, regression_with, regression_without
    )

    # Combine results
    with_mlseakg_df = pd.concat(
        [classification_with, regression_with], ignore_index=True
    )
    without_mlseakg_df = pd.concat(
        [classification_without, regression_without], ignore_index=True
    )

    return with_mlseakg_df, without_mlseakg_df, pd.DataFrame(comparison_data)


def _process_pipeline_baselines(classification_results: list, regression_results: list):
    """Process baseline results for pipeline split mode."""
    baseline_files = glob.glob(
        f"{REMOTE_RESULTS_DIR}/**/baseline_results.json", recursive=True
    )

    for file_path in baseline_files:
        baseline_data = load_json_safe(file_path)
        if not baseline_data:
            continue

        model_dir = os.path.basename(os.path.dirname(file_path))
        model_name_for_plotting = shorten_model_name(model_dir)

        for baseline_type in ["closest_embedding", "average"]:
            if baseline_type not in baseline_data:
                continue

            baseline = baseline_data[baseline_type]
            prefix = "closest" if baseline_type == "closest_embedding" else "average"

            if "accuracy" in baseline:
                classification_results.append(
                    {
                        "model_name": model_dir,
                        "model_name_for_plotting": f"{prefix}-{model_name_for_plotting}",
                        "accuracy": baseline.get("accuracy"),
                        "weighted_f1": baseline.get("weighted_f1"),
                        "params": {},
                    }
                )

            if "mse" in baseline:
                regression_results.append(
                    {
                        "model_name": model_dir,
                        "model_name_for_plotting": f"{prefix}-{model_name_for_plotting}",
                        "mse": baseline.get("mse"),
                        "r2": baseline.get("r2"),
                        "params": {},
                    }
                )


def _process_dataset_baselines(classification_results: list, regression_results: list):
    """Process baseline results for dataset split mode."""
    if not BASELINE_DATASET_SPLIT_RESULTS_PATH.exists():
        return

    baseline_data = load_json_safe(BASELINE_DATASET_SPLIT_RESULTS_PATH)
    if not baseline_data:
        return

    for target_metric, groups in baseline_data.items():
        if not target_metric.startswith("target_metric_"):
            continue

        target = target_metric.replace("target_metric_", "")
        for metafeature_group, models in groups.items():
            if "RF_metamodel" in models:
                clf = models["RF_metamodel"]
                classification_results.append(
                    {
                        "model_name": f"baseline_target_{target}_{metafeature_group}_RF_metamodel",
                        "model_name_for_plotting": f"{metafeature_group}-multiple",
                        "accuracy": clf.get("accuracy"),
                        "weighted_f1": clf.get("weighted_f1"),
                        "params": {},
                    }
                )

            if "RFReg_metamodel" in models:
                reg = models["RFReg_metamodel"]
                regression_results.append(
                    {
                        "model_name": f"baseline_target_{target}_{metafeature_group}_RFReg_metamodel",
                        "model_name_for_plotting": f"{metafeature_group}-multiple",
                        "mse": reg.get("mse"),
                        "r2": reg.get("r2"),
                        "params": {},
                    }
                )


def _create_comparison_data(clf_with, clf_without, reg_with, reg_without):
    """Create comparison data for classification and regression metrics."""
    comparison_data = []

    # Classification comparison
    for metric in ["accuracy", "weighted_f1"]:
        best_with = clf_with[metric].max() if not clf_with.empty else 0
        best_without = clf_without[metric].max() if not clf_without.empty else 0
        improvement = (
            ((best_with - best_without) / best_without * 100) if best_without > 0 else 0
        )

        comparison_data.append(
            {
                "Task": "Classification",
                "Metric": metric.replace("_", " ").title(),
                "Best with MLSeaKG": best_with,
                "Best without MLSeaKG": best_without,
                "Improvement (%)": improvement,
            }
        )

    # Regression comparison
    for metric in ["mse", "r2"]:
        if metric == "mse":
            best_with = reg_with[metric].min() if not reg_with.empty else float("inf")
            best_without = (
                reg_without[metric].min() if not reg_without.empty else float("inf")
            )
            improvement = (
                ((best_without - best_with) / best_without * 100)
                if best_without > 0
                else 0
            )
        else:
            best_with = reg_with[metric].max() if not reg_with.empty else 0
            best_without = reg_without[metric].max() if not reg_without.empty else 0
            improvement = (
                ((best_with - best_without) / best_without * 100)
                if best_without > 0
                else 0
            )

        comparison_data.append(
            {
                "Task": "Regression",
                "Metric": metric.upper(),
                "Best with MLSeaKG": best_with,
                "Best without MLSeaKG": best_without,
                "Improvement (%)": improvement,
            }
        )

    return comparison_data


def main():
    parser = argparse.ArgumentParser(description="MLSeaKG Ablation Study Analysis")
    parser.add_argument(
        "--excl_metrics_with_k",
        nargs="+",
        help="Exclude metrics with k value (e.g., 10 15 20 0.8)",
    )
    parser.add_argument(
        "--excl_dot_product", action="store_true", help="Exclude dot product"
    )
    parser.add_argument(
        "--excl_manhattan", action="store_true", help="Exclude manhattan"
    )
    parser.add_argument(
        "--excl_euclidean", action="store_true", help="Exclude euclidean"
    )
    parser.add_argument("--only_without_mkga", action="store_true", help="Only without MKGA")
    args = parser.parse_args()

    ensure_figures_dir()
    pipeline_prefix = (
        f"mlseakg_ablation_pipeline_{SPLIT_MODE}_min_samples_{MIN_TRAIN_SAMPLES}"
    )
    similarity_prefix = "mlseakg_ablation_similarity"

    # Analyze dataset similarity results
    try:
        sim_with_mlseakg, sim_without_mlseakg = analyze_dataset_similarity_results(
            excl_metrics_with_k=args.excl_metrics_with_k,
            excl_dot_product=args.excl_dot_product,
            excl_manhattan=args.excl_manhattan,
            excl_euclidean=args.excl_euclidean,
            only_without_mkga=args.only_without_mkga,
            only_with_mlsea=False,  # For ablation study, we need both
            selection_metric="NDCG@5",  # Use single metric for selection
        )

        # Save combined similarity results in a single table (similar to pipeline format)
        save_combined_similarity_latex_table(
            sim_with_mlseakg,
            sim_without_mlseakg,
            f"{similarity_prefix}_comparison.tex",
            "Dataset Similarity Results: MLSeaKG Ablation Study",
            metrics_to_show=None,  # None means show all metrics
        )

    except Exception as e:
        print(f"Error analyzing dataset similarity results: {e}")

    # Analyze pipeline performance prediction results
    pipe_comparison = pd.DataFrame()
    try:
        pipe_with_mlseakg, pipe_without_mlseakg, pipe_comparison = (
            analyze_pipeline_performance_prediction()
        )

        if not pipe_comparison.empty:
            # Helper to select best configs and save tables
            def process_and_save_task_results(
                df_with, df_without, task_type, metrics_columns
            ):
                """Process and save results for a specific task type."""
                df_with_task = safe_df(
                    df_with[df_with[metrics_columns[0]].notna()].copy()
                )
                df_without_task = safe_df(
                    df_without[df_without[metrics_columns[0]].notna()].copy()
                )

                # Select best configs
                def select_best(df, with_mlseakg):
                    if df.empty:
                        return df
                    filtered = filter_df_by_mlseakg(df, with_mlseakg)
                    if filtered.empty:
                        return filtered

                    best_rows = []
                    metric = (
                        metrics_columns[1]
                        if len(metrics_columns) > 1
                        else metrics_columns[0]
                    )
                    use_min = task_type == "regression" and metric == "mse"

                    for target in FILTER_TARGETS:
                        sub = filtered[filtered["model_name"].str.contains(target)]
                        if not sub.empty:
                            best_idx = (
                                sub[metric].idxmin()
                                if use_min
                                else sub[metric].idxmax()
                            )
                            best_rows.append(sub.loc[best_idx])
                    return pd.DataFrame(best_rows)

                df_with_best = select_best(df_with_task, True)
                df_without_best = select_best(df_without_task, False)

                if not df_with_best.empty or not df_without_best.empty:
                    # Keep only relevant metrics
                    cols = ["model_name", "model_name_for_plotting"] + metrics_columns
                    df_with_clean = (
                        df_with_best[cols].drop_duplicates()
                        if not df_with_best.empty
                        else pd.DataFrame()
                    )
                    df_without_clean = (
                        df_without_best[cols].drop_duplicates()
                        if not df_without_best.empty
                        else pd.DataFrame()
                    )

                    save_combined_latex_table(
                        df_with_clean,
                        df_without_clean,
                        f"{pipeline_prefix}_{task_type}_comparison.tex",
                        f"{task_type.title()} Results: MLSeaKG Ablation Study",
                        task_type=task_type,
                    )

            # Process classification and regression
            process_and_save_task_results(
                pipe_with_mlseakg,
                pipe_without_mlseakg,
                "classification",
                ["accuracy", "weighted_f1"],
            )
            process_and_save_task_results(
                pipe_with_mlseakg, pipe_without_mlseakg, "regression", ["mse", "r2"]
            )

    except Exception as e:
        print(f"Error analyzing pipeline performance prediction results: {e}")
        print("Full stacktrace:")
        traceback.print_exc()

    print(f"All results saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
