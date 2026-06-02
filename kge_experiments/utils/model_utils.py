# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0


import torch
from typing import Any
from gensim.models.word2vec import Word2Vec as W2V

from kge_experiments.config import RDF2VEC_MODEL_PATH


def load_pykeen_model(model_path: str) -> Any:
    """Load the trained model from the given path."""
    return torch.load(model_path, map_location="cpu")


def load_pykeen_lp_model(model_dir: str, device: str = "cpu") -> Any:
    """
    Load a trained PyKEEN link prediction model.
    
    Args:
        model_dir: Directory containing the trained model (trained_model.pkl)
        device: Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        Loaded PyKEEN model
    """
    import pathlib
    model_path = pathlib.Path(model_dir) / "trained_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    return torch.load(model_path, map_location=device)


def load_rdf2vec_model(
    walk_distance: int, num_walks: int, walk_strategy: str, use_mlseakg: bool, use_mkga: bool
) -> Any:
    """Load the trained model from the given path."""
    return W2V.load(
        str(RDF2VEC_MODEL_PATH).format(
            d=walk_distance,
            w=num_walks,
            ws=walk_strategy,
            kg=("exekgs+mlseakg" if use_mlseakg else "exekgs") + ("_mkga" if use_mkga else ""),
        )
    )
