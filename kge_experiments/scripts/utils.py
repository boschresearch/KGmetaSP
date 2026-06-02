# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

"""
Common utility functions shared across analysis and plotting scripts.
"""

import json
import pandas as pd
from typing import Optional, List


# Constants
BASELINES_END_INDEX = 6


# ============================================================================
# Data Loading and Validation Utilities
# ============================================================================

def load_json_safe(file_path: str) -> Optional[dict]:
    """Load JSON file with error handling."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        print(f"Warning: Could not process {file_path}: {e}")
        return None


def safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return empty DataFrame if input is empty, otherwise return input."""
    return df if not df.empty else pd.DataFrame()


# ============================================================================
# DataFrame Filtering Utilities
# ============================================================================

def filter_df_by_mlseakg(df: pd.DataFrame, with_mlseakg: bool) -> pd.DataFrame:
    """Filter DataFrame based on MLSeaKG usage."""
    if df.empty:
        return df
    flag = "mls_True" if with_mlseakg else "mls_False"
    return df[df["model_name_for_plotting"].str.contains(flag)].copy()


def filter_df(
    df,
    excl_hit_metrics=False,
    excl_metrics_with_k=None,
    excl_dot_product=True,
    excl_manhattan=True,
    excl_euclidean=True,
    only_with_mkga=False,
    only_without_mkga=False,
    only_with_mlsea=True,
):
    """
    Filter DataFrame based on various criteria.
    
    Args:
        df: Input DataFrame
        excl_hit_metrics: Exclude Hit metrics
        excl_metrics_with_k: List of k values to exclude from metrics
        excl_dot_product: Exclude dot product similarity
        excl_manhattan: Exclude manhattan distance
        excl_euclidean: Exclude euclidean distance
        only_with_mkga: Only include models with MKGA preprocessing
        only_without_mkga: Only include models without MKGA preprocessing
        only_with_mlsea: Only include models with MLSeaKG
        
    Returns:
        Filtered DataFrame
    """
    if excl_hit_metrics:
        matching_columns = df.columns[df.columns.str.contains("Hit")]
        df = df.drop(columns=matching_columns)

    if excl_metrics_with_k:
        pattern = "|".join(excl_metrics_with_k)
        matching_columns = df.columns[df.columns.str.contains(pattern)]
        df = df.drop(columns=matching_columns)

    if excl_dot_product:
        matching_indices = df.index[df.index.str.contains("dot")]
        df = df.drop(matching_indices)

    if excl_euclidean:
        matching_indices = df.index[df.index.str.contains("euclidian")]
        df = df.drop(matching_indices)

    if excl_manhattan:
        matching_indices = df.index[df.index.str.contains("manhattan")]
        df = df.drop(matching_indices)

    if only_with_mkga:
        matching_indices = df.index[~df.index.str.contains("mkga")]
        df = df.drop(matching_indices[BASELINES_END_INDEX:])

    if only_without_mkga:
        matching_indices = df.index[df.index.str.contains("mkga")]
        df = df.drop(matching_indices[BASELINES_END_INDEX:])

    if only_with_mlsea:
        matching_indices = df.index[~df.index.str.contains("mlsea")]
        df = df.drop(matching_indices[BASELINES_END_INDEX:])

    return df


def filter_target(df: pd.DataFrame, targets: List[str]) -> pd.DataFrame:
    """
    Filter DataFrame to keep only rows with specified target metrics.
    
    Args:
        df: DataFrame with results
        targets: List of target metrics to keep (e.g., ["accuracy", "precision"])
        
    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df
    # Convert model_name to string to avoid AttributeError on non-string values
    mask = df["model_name"].astype(str).str.contains("|".join(targets))
    return df[mask].reset_index(drop=True)


# ============================================================================
# Model Name Processing Utilities
# ============================================================================

def shorten_model_name(model_name: str) -> str:
    """
    Shortens a folder name by extracting key components for concise plotting.
    Handles both legacy format (emb_source) and new format (dataset_emb, pipeline_emb).
    Always returns format: {dataset_emb}+{pipeline_emb}-mls_{use_mlseakg}-mkga_{use_mkga}
    
    Args:
        model_name: The original folder name
        
    Returns:
        A shortened version of the folder name
    """
    if not isinstance(model_name, str):
        return str(model_name)

    parts = model_name.split("_")

    # Check if we have enough parts for the expected format
    if len(parts) < 15:
        return model_name  # Return original if format doesn't match

    try:
        # Extract key components
        target = parts[2]  # e.g., "accuracy"
        model = parts[4]  # e.g., "RandomForestRegressor"
        emb_aggr = parts[7]  # e.g., "concat"

        # Check if this is the new format (dataset_emb + pipeline_emb) or legacy (emb_source)
        # New format has "dataset" and "emb" at positions 9 and 10
        if len(parts) > 10 and parts[8] == "dataset" and parts[9] == "emb":
            # New format: extract dataset_emb and pipeline_emb
            dataset_emb_parts = []
            pipeline_emb_parts = []
            index_pipeline_emb = 0
            
            # Find dataset_emb (from position 11 until "pipeline")
            for i, part in enumerate(parts[10:], start=10):
                if part == "pipeline":
                    index_pipeline_emb = i + 2  # Skip "pipeline" and "emb"
                    break
                dataset_emb_parts.append(part)
            
            # Find pipeline_emb (from index_pipeline_emb until "mlseakg")
            index_next_to_pipeline_emb = 0
            for i, part in enumerate(parts[index_pipeline_emb:], start=index_pipeline_emb):
                if part == "mlseakg":
                    index_next_to_pipeline_emb = i
                    break
                pipeline_emb_parts.append(part)
            
            if not dataset_emb_parts or not pipeline_emb_parts:
                return model_name
            
            dataset_emb = "_".join(dataset_emb_parts)
            pipeline_emb = "_".join(pipeline_emb_parts)
            
            # Keep dataset_emb and pipeline_emb as-is so individual LP models
            # (TransE, DistMult, ComplEx) are reported separately.
            # Normalize Gated vs non-Gated variants to the same base model name
            # so filter_best_rows groups them and picks the single best.
            if "pykeen" in pipeline_emb:
                for suffix in ("ReaLitEGated", "ReaLitE"):
                    if pipeline_emb.endswith(suffix):
                        pipeline_emb = pipeline_emb[: -len(suffix)]
                        break
            
            # Check if we have enough parts after mlseakg
            if index_next_to_pipeline_emb + 3 >= len(parts):
                return model_name
            
            use_mlseakg = parts[index_next_to_pipeline_emb + 1]
            use_mkga = parts[index_next_to_pipeline_emb + 3]
            
            return f"{dataset_emb}+{pipeline_emb}-mls_{use_mlseakg}-mkga_{use_mkga}"
        else:
            # Legacy format: extract emb_source until "mlseakg"
            # In legacy format, emb_source is the dataset embedding, and pipeline is always rdf2vec
            emb_source_parts = []
            index_next_to_emb_source = 0
            for i, part in enumerate(parts[10:]):
                if part == "mlseakg":
                    index_next_to_emb_source = i + 10
                    break
                emb_source_parts.append(part)

            if not emb_source_parts:  # If no mlseakg found
                return model_name

            dataset_emb = "_".join(emb_source_parts)
            # Simplify pykeen-lp to just show model type
            if "pykeen" in dataset_emb and "lp" in dataset_emb:
                dataset_emb = "pykeen_lp"

            # Check if we have enough parts after mlseakg
            if index_next_to_emb_source + 3 >= len(parts):
                return model_name

            use_mlseakg = parts[index_next_to_emb_source + 1]
            use_mkga = parts[index_next_to_emb_source + 3]

            # For legacy format, pipeline embedding is always rdf2vec
            pipeline_emb = "rdf2vec"
            
            return f"{dataset_emb}+{pipeline_emb}-mls_{use_mlseakg}-mkga_{use_mkga}"
    except (IndexError, ValueError):
        return model_name  # Return original if parsing fails


def extract_bool_flag(model_name_for_plotting: str, flag: str) -> str:
    """
    Extract the value of mlsea or mkga from model_name_for_plotting.
    
    Args:
        model_name_for_plotting: Model name string
        flag: Flag to extract ('mls' or 'mkga')
        
    Returns:
        The value of the flag as a string
    """
    for part in model_name_for_plotting.split("-"):
        if part.startswith(f"{flag}_"):
            return part.split("_")[1]
    return "False"


def extract_emb_source(model_name_for_plotting: str) -> str:
    """
    Extract embedding source from model_name_for_plotting.
    
    Args:
        model_name_for_plotting: Model name string
        
    Returns:
        The embedding source part before '-mls_'
    """
    if "-mls_" in model_name_for_plotting:
        return model_name_for_plotting.split("-mls_")[0]
    elif "-multiple" in model_name_for_plotting:
        return model_name_for_plotting.split("-multiple")[0]
    return model_name_for_plotting


def extract_target(model_name: str) -> str:
    """
    Extract target from model name.
    
    Args:
        model_name: The full model name
        
    Returns:
        Target metric name (e.g., "accuracy")
    """
    if not isinstance(model_name, str):
        return "unknown"
    parts = model_name.split("_")
    for i, part in enumerate(parts):
        if part == "target" and i + 1 < len(parts):
            return parts[i + 1]
    # Fallback: assume target is at position 2
    if len(parts) > 2:
        return parts[2]
    return "unknown"


# ============================================================================
# String Prettification Utilities
# ============================================================================

def prettify_emb_source(s: str) -> str:
    """
    Prettify embedding source names for display.
    
    Args:
        s: Embedding source string
        
    Returns:
        Prettified string
    """
    if s == "pykeen-lp" or s.startswith("pykeen_lp"):
        return "PyKEEN LP"
    s = s.replace("metafeatures", "MF").replace("_", " ")
    return s.title() if not s.startswith("MF") else "MF" + s[2:].title()


def prettify_pipeline_embedding(s: str) -> str:
    """
    Prettify pipeline embedding names for display.
    
    Args:
        s: Pipeline embedding string
        
    Returns:
        Prettified string
    """
    if s == "multiple":
        return "Per-pipeline meta-model"
    elif s == "rdf2vec":
        return "RDF2Vec embedding"
    elif "pykeen" in s and "model" in s:
        # e.g. 'pykeen-lp_pykeen_lp_model_TransEReaLitE' or 'pykeen_lp_model_TransEReaLitE'
        model_part = s.split("model_")[-1] if "model_" in s else s
        for suffix in ("ReaLitEGated", "ReaLitE"):
            if model_part.endswith(suffix):
                model_part = model_part[: -len(suffix)]
                break
        return model_part if model_part else s
    elif s.startswith("pykeen_lp_"):
        # e.g. 'pykeen_lp_TransEReaLitE' -> 'TransE'
        model_part = s[len("pykeen_lp_"):]
        for suffix in ("ReaLitEGated", "ReaLitE"):
            if model_part.endswith(suffix):
                model_part = model_part[: -len(suffix)]
                break
        return model_part if model_part else s
    elif s == "pykeen-lp" or s == "pykeen_lp":
        return "PyKEEN LP embedding"
    else:
        return s.replace("_", " ")


def prettify_metric(s: str) -> str:
    """
    Prettify metric names for display.
    
    Args:
        s: Metric name string
        
    Returns:
        Prettified string
    """
    return s.replace("weighted_", "").replace("_", " ").title()


def prettify_target(s: str) -> str:
    """
    Prettify target names for display.
    
    Args:
        s: Target name string
        
    Returns:
        Prettified string
    """
    return s.replace("_", " ").title()


# ============================================================================
# LaTeX Formatting Utilities
# ============================================================================

def latex_bold(text: str) -> str:
    """
    Wrap text in LaTeX bold formatting.
    
    Args:
        text: Text to make bold
        
    Returns:
        LaTeX formatted bold text
    """
    return f"\\textbf{{{text}}}"


def strip_textbf(s: str) -> str:
    """
    Remove LaTeX bold formatting from text.
    
    Args:
        s: Text potentially containing \\textbf{} formatting
        
    Returns:
        Text without bold formatting
    """
    if isinstance(s, str) and s.startswith(r"\textbf{") and s.endswith("}"):
        return s[len(r"\textbf{"):-1]
    return s


def distinguish_duplicate_indices_latex(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add LaTeX superscript numbering to duplicate index values.
    
    Args:
        df: DataFrame with potentially duplicate indices
        
    Returns:
        DataFrame with distinguished indices
    """
    if df.index.is_unique:
        return df.copy()  # No duplicates, return original DataFrame

    new_index = []
    counts = {}

    for index_val in df.index:
        if index_val not in counts:
            counts[index_val] = 1
            new_index.append(index_val)
        else:
            counts[index_val] += 1
            new_index.append(f"{index_val}$\\!^{counts[index_val]}$")

    df = df.copy()  # Important: Copy the original df, to prevent changing the original.
    df.index = new_index
    return df
