# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import glob
import os
import json
import pandas as pd
import pathlib
from kge_experiments.config import DATA_DIR
from kge_experiments.scripts.utils import (
    shorten_model_name,
    extract_emb_source,
    extract_target,
    prettify_emb_source,
    prettify_pipeline_embedding,
    prettify_target,
    filter_target,
)

# Define the base directory for results
MIN_TRAIN_SAMPLES = int(os.environ.get("MIN_TRAIN_SAMPLES", 50))

# Define the base directory for results
# Legacy pattern (with emb_source)
REMOTE_RESULTS_DIR_DATASET_LEGACY = (
    pathlib.Path(
        "/fs/scratch/rb_bd_dlp_rng-dl01_cr_AIQ_employees/students/klr2rng/perf_pred_results"
    )
    / f"results_target_*_emb_source_*_split_dataset_mintrainsamples_{MIN_TRAIN_SAMPLES}"
)
REMOTE_RESULTS_DIR_PIPELINE_LEGACY = (
    pathlib.Path(
        "/fs/scratch/rb_bd_dlp_rng-dl01_cr_AIQ_employees/students/klr2rng/perf_pred_results"
    )
    / f"results_target_*_emb_source_*_split_pipeline_mintrainsamples_{MIN_TRAIN_SAMPLES}"
)

# New pattern (with dataset_emb and pipeline_emb)
REMOTE_RESULTS_DIR_DATASET_NEW = (
    pathlib.Path(
        "/fs/scratch/rb_bd_dlp_rng-dl01_cr_AIQ_employees/students/klr2rng/perf_pred_results"
    )
    / f"results_target_*_dataset_emb_*_pipeline_emb_*_split_dataset_mintrainsamples_{MIN_TRAIN_SAMPLES}"
)
REMOTE_RESULTS_DIR_PIPELINE_NEW = (
    pathlib.Path(
        "/fs/scratch/rb_bd_dlp_rng-dl01_cr_AIQ_employees/students/klr2rng/perf_pred_results"
    )
    / f"results_target_*_dataset_emb_*_pipeline_emb_*_split_pipeline_mintrainsamples_{MIN_TRAIN_SAMPLES}"
)

LOCAL_RESULTS_DIR = DATA_DIR / "perf_pred_results"
BASELINE_DATASET_SPLIT_RESULTS_PATH = (
    LOCAL_RESULTS_DIR / f"analysis_results_min_{MIN_TRAIN_SAMPLES}_samples.json"
)

# Add config for which split mode to use
SPLIT_MODE = os.environ.get("SPLIT_MODE", "dataset")  # or "pipeline"

if SPLIT_MODE == "pipeline":
    REMOTE_RESULTS_DIRS = [REMOTE_RESULTS_DIR_PIPELINE_LEGACY, REMOTE_RESULTS_DIR_PIPELINE_NEW]
else:
    REMOTE_RESULTS_DIRS = [REMOTE_RESULTS_DIR_DATASET_LEGACY, REMOTE_RESULTS_DIR_DATASET_NEW]


FILTER_TARGETS = ["accuracy", "precision"]


def save_combined_latex_table(clf_df, reg_df, filename):
    """
    Save a single merged LaTeX table combining regression (MSE, R2) and
    classification (Accuracy, F1) metrics.  Rows are grouped by target metric
    with a \\multicolumn target-header row.  Layout depends on SPLIT_MODE:
    - dataset: Dataset Emb. | Pipeline Strategy | MSE | R2 | Accuracy | F1
    - pipeline: Method | MSE | R2 | Accuracy | F1
    """
    if clf_df.empty and reg_df.empty:
        print("No data for combined table.")
        return

    def _get_emb_source(x):
        return x.split("+")[0] if "+" in x else x.split("-multiple")[0]

    def _get_pipeline_emb(x):
        if "+" not in x:
            if x.endswith("-multiple"):
                return "multiple"
            if x.startswith("average") or x.startswith("closest"):
                return x.split("-")[0]
            return "rdf2vec"
        after = x.split("+", 1)[1]
        if after.startswith("rdf2vec"):
            return "rdf2vec"
        return after  # e.g. 'pykeen-lp_pykeen_lp_model_TransE'

    def _pipeline_sort_key(pe):
        if pe in ("multiple", "average", "closest"):
            return 0
        if "pykeen" in pe:
            # stable ordering by model name
            return 10 + hash(pe) % 10
        if pe == "rdf2vec":
            return 99
        return 50

    clf = clf_df.copy()
    reg = reg_df.copy()

    for df in (clf, reg):
        df["emb_source"] = df["model_name_for_plotting"].apply(_get_emb_source)
        df["pipeline_emb"] = df["model_name_for_plotting"].apply(_get_pipeline_emb)
        df["target"] = df["model_name"].apply(extract_target)

    merged = pd.merge(
        clf[["emb_source", "pipeline_emb", "target", "accuracy", "weighted_f1"]],
        reg[["emb_source", "pipeline_emb", "target", "mse", "r2"]],
        on=["emb_source", "pipeline_emb", "target"],
        how="outer",
    )

    # For pipeline split: collapse baseline variants
    # - "average*" rows are all identical → keep one per target
    # - "closest*" rows → keep best per target (highest accuracy; tiebreak on lowest mse)
    if SPLIT_MODE == "pipeline":
        non_baselines = merged[~(merged["emb_source"].str.startswith("average") |
                                  merged["emb_source"].str.startswith("closest"))].copy()
        avg_rows = merged[merged["emb_source"].str.startswith("average")].copy()
        clo_rows = merged[merged["emb_source"].str.startswith("closest")].copy()

        avg_dedup = avg_rows.drop_duplicates(subset=["target", "accuracy", "weighted_f1", "mse", "r2"])
        avg_dedup["emb_source"] = "average"

        best_closest = []
        for tgt in clo_rows["target"].unique():
            sub = clo_rows[clo_rows["target"] == tgt].copy()
            sub["_acc"] = pd.to_numeric(sub["accuracy"], errors="coerce")
            sub["_mse"] = pd.to_numeric(sub["mse"], errors="coerce")
            best = sub.sort_values(["_acc", "_mse"], ascending=[False, True]).iloc[0].drop(["_acc", "_mse"])
            best["emb_source"] = "closest"
            best_closest.append(best.to_dict())
        clo_dedup = pd.DataFrame(best_closest) if best_closest else pd.DataFrame()

        merged = pd.concat([avg_dedup, clo_dedup, non_baselines], ignore_index=True)

    METRICS = {
        "mse": "min",
        "r2": "max",
        "accuracy": "max",
        "weighted_f1": "max",
    }

    # Return formatted string for a cell value
    def fmt(val, col, is_group_best, is_overall_best):
        fval = pd.to_numeric(val, errors="coerce")
        if pd.isnull(fval):
            return ""
        s = f"{fval:.4f}"
        if is_overall_best:
            s = r"\underline{" + s + "}"
        if is_group_best and SPLIT_MODE == "dataset":
            s = r"\textbf{" + s + "}"
        return s

    def method_label(emb_source, pipeline_emb):
        base = emb_source.split("-")[0]
        if base in ("average", "closest"):
            return "Average performance" if base == "average" else "Closest embedding"
        es_pretty = prettify_emb_source(emb_source)
        pe_pretty = prettify_pipeline_embedding(pipeline_emb)
        return f"{es_pretty} + {pe_pretty}"

    if SPLIT_MODE == "dataset":
        n_cols = 6
        col_format = "@{}llcccc@{}"
        header = (
            r"\textbf{Dataset} & \textbf{Pipeline} & \multicolumn{2}{c}{\textbf{Meta-Regr.}} & \multicolumn{2}{c}{\textbf{Meta-Classif.}} \\"
            + "\n"
            + r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}"
            + "\n"
            + r"\textbf{Emb.} & \textbf{Strategy} & \textbf{MSE} & \textbf{R$^2$} & \textbf{Accuracy} & \textbf{F1} \\"
        )
    else:
        n_cols = 5
        col_format = "lcccc"
        header = (
            r"\textbf{Method} & \multicolumn{2}{c}{\textbf{Meta-Regr.}} & \multicolumn{2}{c}{\textbf{Meta-Classif.}} \\"
            + "\n"
            + r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}"
            + "\n"
            + r"& \textbf{MSE} & \textbf{R$^2$} & \textbf{Accuracy} & \textbf{F1} \\"
        )

    lines = [
        r"\begin{tabular}{" + col_format + "}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    targets = [t for t in FILTER_TARGETS if t in merged["target"].values]
    for ti, target in enumerate(targets):
        tdf = merged[merged["target"] == target].copy()
        if tdf.empty:
            continue

        tdf["_psort"] = tdf["pipeline_emb"].apply(_pipeline_sort_key)
        tdf = tdf.sort_values(["emb_source", "_psort", "pipeline_emb"]).reset_index(drop=True)

        # Per-target overall bests across all rows
        overall_best = {}
        for col, mode in METRICS.items():
            if col not in tdf.columns:
                continue
            vals = pd.to_numeric(tdf[col], errors="coerce").dropna()
            overall_best[col] = (vals.min() if mode == "min" else vals.max()) if len(vals) else None

        # Per-target, per-emb_source bests
        group_best = {}
        for es in tdf["emb_source"].unique():
            sub = tdf[tdf["emb_source"] == es]
            group_best[es] = {}
            for col, mode in METRICS.items():
                if col not in sub.columns:
                    continue
                vals = pd.to_numeric(sub[col], errors="coerce").dropna()
                group_best[es][col] = (vals.min() if mode == "min" else vals.max()) if len(vals) else None

        lines.append(
            r"\multicolumn{" + str(n_cols) + r"}{l}{\textbf{Target: "
            + prettify_target(target)
            + r"}} \\"
        )

        prev_emb_source = None
        for _, row in tdf.iterrows():
            es = row["emb_source"]
            pe = row["pipeline_emb"]

            # thin separator between emb_source groups (dataset split only)
            if SPLIT_MODE == "dataset" and prev_emb_source is not None and es != prev_emb_source:
                lines.append(r"\cmidrule(l){2-6}")
            prev_emb_source = es

            cells = []
            for col in ("mse", "r2", "accuracy", "weighted_f1"):
                val = row.get(col, float("nan"))
                fval = pd.to_numeric(val, errors="coerce")
                ob = overall_best.get(col)
                gb = group_best.get(es, {}).get(col)
                cells.append(fmt(val, col, (gb is not None and not pd.isnull(fval) and fval == gb),
                                          (ob is not None and not pd.isnull(fval) and fval == ob)))

            if SPLIT_MODE == "dataset":
                es_p = prettify_emb_source(es)
                pe_p = prettify_pipeline_embedding(pe)
                lines.append(f"{es_p} & {pe_p} & {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} \\\\")
            else:
                lines.append(f"{method_label(es, pe)} & {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} \\\\")

        if ti < len(targets) - 1:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}"]

    latex_str = "\n".join(lines)
    with open(os.path.join(LOCAL_RESULTS_DIR, filename), "w") as f:
        f.write(latex_str)
    print(f"Combined LaTeX table saved to {filename}")


# Define the combinations to keep
COMBINATIONS = [
    {"mlsea": "False", "mkga": "False"},
    {"mlsea": "True", "mkga": "False"},
    {"mlsea": "True", "mkga": "True"},
]


def filter_best_rows(df, task, baseline_only=False):
    """
    Finds the best performing instance for each (emb_source, target) pair.
    If baseline_only is True, only considers baseline rows.
    If baseline_only is False, only considers non-baseline rows.
    """
    if df.empty:
        return df
    df = df.copy()

    if baseline_only:
        df = df[
            (df["model_name_for_plotting"].str.startswith("average"))
            | (df["model_name_for_plotting"].str.startswith("closest"))
        ]
    else:
        df = df[
            (~df["model_name_for_plotting"].str.startswith("average"))
            & (~df["model_name_for_plotting"].str.startswith("closest"))
            & (~df["model_name_for_plotting"].str.endswith("-multiple"))
        ]

    # Use extract_emb_source and extract_target from common_utils
    df["emb_source"] = df["model_name_for_plotting"].apply(extract_emb_source)
    df["target"] = df["model_name"].apply(extract_target)
    keep_rows = []
    for emb_source in df["emb_source"].unique():
        for target in df["target"].unique():
            sub = df[(df["emb_source"] == emb_source) & (df["target"] == target)]
            if sub.empty:
                continue
            if task == "classification":
                idx = sub["weighted_f1"].idxmax()
            else:
                idx = sub["mse"].idxmin()
            row = df.loc[idx].copy()
            # For non-baseline, replace everything starting from '-mls' with '+rdf2vec'
            # if not baseline_only:
            row["model_name_for_plotting"] = (
                row["model_name_for_plotting"].split("-mls")[0] #+ "+rdf2vec"
            )
            keep_rows.append(row.to_dict())
    return pd.DataFrame(keep_rows)


if __name__ == "__main__":
    if not LOCAL_RESULTS_DIR.exists():
        LOCAL_RESULTS_DIR.mkdir(parents=True)
        print(f"Created local results directory at {LOCAL_RESULTS_DIR}")
        
    # Initialize lists to store results
    classification_results = []
    regression_results = []

    # Use glob to find all JSON files under both legacy and new directories
    json_files = []
    baseline_json_files = []
    
    for REMOTE_RESULTS_DIR in REMOTE_RESULTS_DIRS:
        json_files.extend(glob.glob(
            f"{REMOTE_RESULTS_DIR}/**/best_model_results.json", recursive=True
        ))
        # Also include baseline_results.json if present (for pipeline split)
        baseline_json_files.extend(glob.glob(
            f"{REMOTE_RESULTS_DIR}/**/baseline_results.json", recursive=True
        ))

    print(f"Found {len(json_files)} result files")
    print(f"Found {len(baseline_json_files)} baseline files")

    for file_path in json_files:
        with open(file_path, "r") as f:
            data = json.load(f)
            # Extract task type from the file path
            if "accuracy" in data.get("test_metrics", {}):  # Classification task
                classification_results.append(
                    {
                        "model_name": os.path.basename(os.path.dirname(file_path)),
                        "model_name_for_plotting": shorten_model_name(
                            os.path.basename(os.path.dirname(file_path))
                        ),
                        "accuracy": data["test_metrics"].get("accuracy"),
                        "weighted_f1": data["test_metrics"].get("weighted_f1"),
                        "params": data.get("best_model_params", {}),
                    }
                )
            elif "mse" in data.get("test_metrics", {}):  # Regression task
                regression_results.append(
                    {
                        "model_name": os.path.basename(os.path.dirname(file_path)),
                        "model_name_for_plotting": shorten_model_name(
                            os.path.basename(os.path.dirname(file_path))
                        ),
                        "mse": data["test_metrics"].get("mse"),
                        "r2": data["test_metrics"].get("r2"),
                        "params": data.get("best_model_params", {}),
                    }
                )

    if SPLIT_MODE == "pipeline":
        # --- Add baseline results from baseline_results.json (for pipeline split) ---
        for file_path in baseline_json_files:
            print(f"Processing baseline file: {file_path}")
            with open(file_path, "r") as f:
                baseline_data = json.load(f)
            model_dir = os.path.basename(os.path.dirname(file_path))
            model_name_for_plotting = shorten_model_name(model_dir)
            # Classification baselines
            if "closest_embedding" in baseline_data:
                if "accuracy" in baseline_data["closest_embedding"]:
                    classification_results.append(
                        {
                            "model_name": f"{model_dir}",
                            "model_name_for_plotting": f"closest-{model_name_for_plotting}",
                            "accuracy": baseline_data["closest_embedding"].get(
                                "accuracy"
                            ),
                            "weighted_f1": baseline_data["closest_embedding"].get(
                                "weighted_f1"
                            ),
                            "params": {},
                        }
                    )
                if "accuracy" in baseline_data["average"]:
                    classification_results.append(
                        {
                            "model_name": f"{model_dir}",
                            "model_name_for_plotting": f"average-{model_name_for_plotting}",
                            "accuracy": baseline_data["average"].get("accuracy"),
                            "weighted_f1": baseline_data["average"].get("weighted_f1"),
                            "params": {},
                        }
                    )
            # Regression baselines
            if "closest_embedding" in baseline_data:
                if "mse" in baseline_data["closest_embedding"]:
                    regression_results.append(
                        {
                            "model_name": f"{model_dir}",
                            "model_name_for_plotting": f"closest-{model_name_for_plotting}",
                            "mse": baseline_data["closest_embedding"].get("mse"),
                            "r2": baseline_data["closest_embedding"].get("r2"),
                            "params": {},
                        }
                    )
                if "mse" in baseline_data["average"]:
                    regression_results.append(
                        {
                            "model_name": f"{model_dir}",
                            "model_name_for_plotting": f"average-{model_name_for_plotting}",
                            "mse": baseline_data["average"].get("mse"),
                            "r2": baseline_data["average"].get("r2"),
                            "params": {},
                        }
                    )
    elif SPLIT_MODE == "dataset":
        # --- Add baseline results from analysis_results.json ---
        if BASELINE_DATASET_SPLIT_RESULTS_PATH.exists():
            with open(BASELINE_DATASET_SPLIT_RESULTS_PATH, "r") as f:
                baseline_data = json.load(f)
            # For each target metric (e.g., f1_score, accuracy, precision)
            for target_metric, groups in baseline_data.items():

                if not target_metric.startswith("target_metric_"):
                    continue

                target = target_metric.replace("target_metric_", "")
                for metafeature_group, models in groups.items():
                    # Classification baseline
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
                    # Regression baseline
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

    # Convert results to DataFrames
    classification_df = pd.DataFrame(classification_results)
    regression_df = pd.DataFrame(regression_results)

    classification_df_best = filter_best_rows(
        classification_df, "classification", baseline_only=False
    )

    pd.set_option('display.max_colwidth', 1000)
    print(f"Best classification models: {classification_df_best['model_name']}")

    if SPLIT_MODE == "pipeline":
        classification_baseline_df_best = filter_best_rows(
            classification_df, "classification", baseline_only=True
        )
    else:
        classification_baseline_df_best = classification_df[
            classification_df["model_name_for_plotting"].str.endswith("-multiple")
        ]

    classification_df = pd.concat(
        [classification_df_best, classification_baseline_df_best], ignore_index=True
    )

    regression_df_best = filter_best_rows(
        regression_df, "regression", baseline_only=False
    )
    
    print(f"Best regression models: {regression_df_best['model_name']}")
    
    if SPLIT_MODE == "pipeline":
        regression_baseline_df_best = filter_best_rows(
            regression_df, "regression", baseline_only=True
        )
    else:
        regression_baseline_df_best = regression_df[
            regression_df["model_name_for_plotting"].str.endswith("-multiple")
        ]

    regression_df = pd.concat(
        [regression_df_best, regression_baseline_df_best], ignore_index=True
    )

    classification_df = filter_target(classification_df, FILTER_TARGETS)
    regression_df = filter_target(regression_df, FILTER_TARGETS)

    print(f"Best classification model: {classification_df['model_name'].iloc[0]}")
    print(f"Best regression model: {regression_df['model_name'].iloc[0]}")

    # Save single combined table (regression + classification per row, grouped by target)
    save_combined_latex_table(
        classification_df,
        regression_df,
        filename=f"combined_results_split_{SPLIT_MODE}_min_samples_{MIN_TRAIN_SAMPLES}_table.tex",
    )
