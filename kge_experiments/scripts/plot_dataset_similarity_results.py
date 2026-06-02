# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import argparse
import pandas as pd
from collections import Counter

from kge_experiments.config import DATA_DIR
from kge_experiments.scripts.utils import (
    filter_df,
    latex_bold,
    distinguish_duplicate_indices_latex,
    BASELINES_END_INDEX,
)

RETRIEVAL_RESULTS_DIR = DATA_DIR / "retrieval_results"
RETRIEVAL_RANKING_RESULTS_PATH = (
    RETRIEVAL_RESULTS_DIR / "perf_results_cosine_sim_excl_perf.csv"
)

FIGURES_DIR = RETRIEVAL_RESULTS_DIR / "figures"


# Define the levels
display_config_names = {
    "RDF2Vec Parameter Sets": {
        "d_10_w_10": "d_10_w_10",
        "d_15_w_10": "d_15_w_10",
        "d_20_w_10": "d_20_w_10",
    },
    "Embedding Type": {
        "data": "FeatSim",
        "pip": "PipSim",
        "comb": "DatSim",
    },
    "MLSeaKG Integration": {"no_mlsea": "-MLSeaKG", "with_mlsea": "+MLSeaKG"},
    "MKGA Preprocessing": {"no_mkga": "-MKGA", "with_mkga": "+MKGA"},
    "Distance Measures": {
        "cosine": "cosine",
        "euclidian": "euclidean",
        "manhattan": "manhattan",
        "dot_product": "dot_product",
    },
}


def parse_levels(name):
    parts = name.split("_")
    param_set = f"d_{parts[2][1:]}_w_{parts[3][1:]}"
    embedding_type = display_config_names["Embedding Type"][parts[0]]
    mlsea_kg = (
        display_config_names["MLSeaKG Integration"]["with_mlsea"]
        if "mlsea" in parts
        else display_config_names["MLSeaKG Integration"]["no_mlsea"]
    )
    mkga = (
        display_config_names["MKGA Preprocessing"]["with_mkga"]
        if "mkga" in parts
        else display_config_names["MKGA Preprocessing"]["no_mkga"]
    )
    similarity_measure = parts[-1] if parts[-1] != "mkga" else parts[-2]
    return param_set, embedding_type, mlsea_kg, mkga, similarity_measure


def save_latex_table(
    df, filename, highlight_max=True, hline_row_index=None, baselines_end_index=0
):
    def highlight_max(s):
        if s.dtype == "object":
            return s
        max_val = s.max()
        second_max_val = s.nlargest(2).iloc[-1]
        return [
            (
                "\\textbf{" + str(v) + "}"
                if v == max_val
                else "\\underline{" + str(v) + "}" if v == second_max_val else str(v)
            )
            for v in s
        ]

    sim_ged = "GED"
    # sim_ged = "$\\text{Sim}_{GED}$"
    df.index = pd.Index(
        list(
            df.index[:baselines_end_index].map(
                lambda name: latex_bold(
                    name.replace("sf_bi_encoder_dataset_metadata", "Dataset-BiEncoder")
                    .replace("sf_bi_encoder_models_metadata", "Model-BiEncoder")
                    .replace("multi_field_bi_encoder", "Multi-BiEncoder")
                    .replace("scaled_p_norm", "Scaled p-Norm")
                    .replace("p_norm", "p-Norm")
                    .replace("ged", sim_ged)
                )
            )
        )
        + list(
            df.index[baselines_end_index:].map(
                lambda name: latex_bold(
                    "".join(name.split("_")[4:])
                    .replace("-MLSeaKG", "")
                    .replace("+MLSeaKG", ",\\textit{MLSea}")
                    .replace("-MKGA", "")
                    .replace("+MKGA", ",\\mkga")
                    # .replace("PipSim", "\\textit{Pip}")
                    # .replace("FeatSim", "\\textit{Feat}")
                    # .replace("DatSim", "\\textit{Dat}")
                    .replace("PipSim", "\\pipsim")
                    .replace("FeatSim", "\\featsim")
                    .replace("DatSim", "\\datsim")
                    # .replace("PipSim", "\\pipsim ")
                    # .replace("FeatSim", "\\featsim ")
                    # .replace("DatSim", "\\datsim ")
                    .replace("cosine", "")
                    .replace("euclidian", "")
                    .replace("manhattan", "")
                    .replace("prod", "")
                    # .replace("cosine", " (cos)")
                    # .replace("euclidian", " (eucl)")
                    # .replace("manhattan", " (manh)")
                    # .replace("prod", "(prod)")
                )
            )
        )
    )

    df.columns = df.columns.map(
        lambda name: f"\\textbf{{{name.replace(' (0.9)', '').replace(' (0.8)', '')}}}"
    )

    if highlight_max:
        df = df.apply(highlight_max)

    # if highlight_second_max:
    #     df = df.apply(highlight_second_max)

    df = pd.concat(
        [
            df.iloc[:baselines_end_index, :],
            df.iloc[baselines_end_index:, :].sort_index(),
        ]
    )

    df = distinguish_duplicate_indices_latex(df)

    df = df.reset_index()
    df = df.rename(columns={"index": "\\textbf{Similarity measure}"})

    latex_table = df.to_latex(
        index=False,
        escape=False,
        float_format="%.4f",
        column_format="l" + "c" * len(df.columns),
    )

    if hline_row_index is not None:
        lines = latex_table.splitlines()
        lines.insert(
            hline_row_index + 4, "\\hline"
        )  # +3 to account for LaTeX table header
        latex_table = "\n".join(lines)

    with open(FIGURES_DIR / filename, "w") as f:
        f.write(latex_table)


def find_best_model_per_embedding_type(df):
    """
    Find the best performing model for each embedding type by counting
    which model wins the most metrics WITHIN that embedding type.
    
    Returns:
        dict: Mapping of embedding type to best model name
    """
    # Parse embedding types for each model and group by type
    models_by_type = {
        "PipSim": [],
        "DatSim": [],
        "FeatSim": [],
    }
    
    for model_name in df.index:
        if "emb" in model_name:
            _, embedding_type, _, _, _ = parse_levels(model_name)
            models_by_type[embedding_type].append(model_name)

    # For each embedding type, find which model wins most metrics within that type
    best_models = {}
    
    for emb_type in ["PipSim", "DatSim", "FeatSim"]:
        if not models_by_type[emb_type]:
            print(f"\n{emb_type}: No models of this type")
            best_models[emb_type] = None
            continue
        
        # Filter to only models of this embedding type
        type_df = df.loc[models_by_type[emb_type]]
        
        # Count wins for each model within this type
        wins_counter = Counter()
        for metric in type_df.columns:
            best_model = type_df[metric].idxmax()
            wins_counter[best_model] += 1
        
        # Get the model with most wins
        best_model, wins = wins_counter.most_common(1)[0]
        best_models[emb_type] = best_model
        print(f"\n{emb_type} best model: {best_model} ({wins}/{len(type_df.columns)} wins)")
        
        # Show all models and their wins for this type
        print(f"  All {emb_type} models:")
        for model, count in wins_counter.most_common():
            param_set, _, mlsea, mkga, sim = parse_levels(model)
            print(f"    - {param_set}, {mlsea}, {mkga}: {count} wins")

    return best_models


def main():
    parser = argparse.ArgumentParser(description="Process exclusion variables.")
    parser.add_argument(
        "--excl_hit_metrics", action="store_true", help="Exclude Hit metrics"
    )
    parser.add_argument(
        "--excl_metrics_with_k",
        nargs="+",
        help="Exclude metrics with k value",
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
    parser.add_argument("--only_with_mkga", action="store_true", help="Only with MKGA")
    parser.add_argument(
        "--only_with_mlsea", action="store_true", help="Only with MLSeaKG"
    )
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    retrieval_rank_results_df = pd.read_csv(RETRIEVAL_RANKING_RESULTS_PATH, index_col=0)

    retrieval_rank_results_df = filter_df(
        retrieval_rank_results_df,
        excl_hit_metrics=args.excl_hit_metrics,
        excl_metrics_with_k=args.excl_metrics_with_k,
        excl_dot_product=args.excl_dot_product,
        excl_manhattan=args.excl_manhattan,
        excl_euclidean=args.excl_euclidean,
        only_with_mkga=args.only_with_mkga,
        only_with_mlsea=args.only_with_mlsea,
    )

    # Find best model per embedding type BEFORE renaming
    only_kge_df_original = retrieval_rank_results_df.iloc[BASELINES_END_INDEX:, :]
    best_models_per_emb_type = find_best_model_per_embedding_type(only_kge_df_original)

    # Rename KGE models to readable format
    multiindex_levels = []
    rename_mapping = {}
    for name in retrieval_rank_results_df.index:
        if "emb" in name:
            config_levels = parse_levels(name)
            new_name = "_".join(config_levels)
            multiindex_levels.append(config_levels)
            rename_mapping[name] = new_name
    
    retrieval_rank_results_df = retrieval_rank_results_df.rename(index=rename_mapping)

    # Get the best model rows for each embedding type
    best_kge_rows = []
    for emb_type in ["PipSim", "DatSim", "FeatSim"]:
        if best_models_per_emb_type[emb_type] is not None:
            # Map original name to renamed name
            renamed_model = rename_mapping.get(best_models_per_emb_type[emb_type])
            if renamed_model:
                best_kge_rows.append(renamed_model)

    # Create table with baselines and best model per embedding type
    baselines_and_best_per_emb_type = pd.concat(
        [
            retrieval_rank_results_df.iloc[:BASELINES_END_INDEX, :],
            retrieval_rank_results_df.loc[best_kge_rows],
        ]
    )

    baselines_and_best_per_emb_type = baselines_and_best_per_emb_type.map(
        lambda v: round(v, 4)
    )

    print("\n" + "="*80)
    print("BASELINES AND BEST MODEL PER EMBEDDING TYPE")
    print("="*80)
    print(baselines_and_best_per_emb_type.to_string())

    save_latex_table(
        baselines_and_best_per_emb_type.copy(),
        "baselines_and_best_per_emb_type.tex",
        hline_row_index=BASELINES_END_INDEX,
        baselines_end_index=BASELINES_END_INDEX,
    )
    
    print(f"\nLaTeX table saved to: {FIGURES_DIR / 'baselines_and_best_per_emb_type.tex'}")


if __name__ == "__main__":
    main()
