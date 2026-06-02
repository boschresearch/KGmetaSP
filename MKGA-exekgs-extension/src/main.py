# This source code is from MKGA (w/ slight adaptations)
#   (https://gitlab.com/patryk.preisner/mkga/-/blob/e5a065403371403bec56686f9425d55a59f4cb1f/src/main.py)
# Copyright (c) 2023 Patryk Preisner
# This source code is licensed under the Apache license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.

"""
Main method for the mdoular GDA apporach evaluation framework

Author: Patryk Preisner
Date: May 31, 2023

This file is the main file to execute for a single evaluation step, otherwise autoevaluate.py should be executed.
"""

from pathlib import Path
import pathlib
import sys

path_root = Path(__file__).parent
sys.path.append(str(path_root))

from omegaconf import DictConfig
import hydra
from utils.data_utils import extract_ents
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import logging
import os
import numpy as np
import torch
import random
import pickle
import preprocess

# internal libraries:
import dataload
import embed
import evaluate
from utils import Data

log = logging.getLogger(__name__)

HERE = pathlib.Path(__file__).parent

BASEPATH = str(HERE / ".." / "data" / "preprocessed")


@hydra.main(version_base=None, config_path="../config", config_name="single")
def evaluate_one(cfg: DictConfig) -> None:
    """
    Evaluate a specific GDA approach under selected embedder and dataset

    Args:
        cfg (DictConfig): hydra config file
    """
    evaluate_approach(cfg)


def preprocess_data(cfg: DictConfig) -> None:
    _setup_seed(42)
    log.info("Data loading...")
    # Load preprocessed .pickle file if exist, else load and preprocess dataset
    if _exist_preprocessed_data(cfg):
        data = _load_preprocessed_data(cfg)
        log.info("Preprocessed Data found, Skipping Preprocess...")
    else:
        data = getattr(dataload, cfg["pipeline"]["dataload"])(
            **cfg["dataload"][cfg["pipeline"]["dataload"]]
        )
        data.name = (
            f'{cfg["pipeline"]["dataload"]}+{cfg["pipeline"]["augment"]}'
        )

        log.info("Preprocess started...")
        for step in cfg["aug_approach"][cfg["pipeline"]["augment"]]:
            log.info(f"Processing step {step}...")
            data = getattr(preprocess, step)(data, **cfg["aug_method"][step])
        _save_preprocessed_data(data, cfg)

    return data


def _load_preprocessed_data(cfg: DictConfig) -> Data:
    """
    Load dataset based on cfg -> pipeline -> dataload variable

    Args:
        cfg (DictConfig): hydra config file
    Returns:
        Data: KG, encoded using kgbench Dataset object
    """
    with open(
        f'{BASEPATH}/{cfg["pipeline"]["dataload"]}+{cfg["pipeline"]["augment"]}.pickle',
        "rb",
    ) as f:
        data: Data = pickle.load(f)
    return data


def _save_preprocessed_data(data: Data, cfg: DictConfig):
    """
    Saves given KG into "..data/preprocessed" as .pickle file

    Args:
        data (Data): KG, encoded using kgbench Dataset object
        cfg (DictConfig): hydra config file
    """
    with open(
        f'{BASEPATH}/{cfg["pipeline"]["dataload"]}+{cfg["pipeline"]["augment"]}.pickle',
        "wb",
    ) as f:
        pickle.dump(data, f)


def _exist_preprocessed_data(cfg: DictConfig):
    """
    Checks if preprocessed .pickle dataset exist under "..data/preprocessed"

    Args:
        cfg (DictConfig): hydra config file
    """
    return os.path.exists(
        f'{BASEPATH}/{cfg["pipeline"]["dataload"]}+{cfg["pipeline"]["augment"]}.pickle'
    )


def _setup_seed(seed: int):
    """
    Sets global seed attributes within numpy, torch, random and PYTHONHASHSEED
    to ensure determinism throughout all python packages used.

    Args:
        seed (int): seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def evaluate_approach(cfg: DictConfig) -> None:
    """
    Main method, that coordinates the evaluation of a specific GDA approach under specific dataset and embedder.

    Args:
        cfg (DictConfig): hydra config file
    """
    _setup_seed(42)
    log.info("Data loading...")
    # Load preprocessed .pickle file if exist, else load and preprocess dataset
    if _exist_preprocessed_data(cfg):
        data = _load_preprocessed_data(cfg)
        log.info("Preprocessed Data found, Skipping Preprocess...")
    else:
        data = getattr(dataload, cfg["pipeline"]["dataload"])(
            **cfg["dataload"][cfg["pipeline"]["dataload"]]
        )
        data.name = (
            f'{cfg["pipeline"]["dataload"]}+{cfg["pipeline"]["augment"]}'
        )

        log.info("Preprocess started...")
        for step in cfg["aug_approach"][cfg["pipeline"]["augment"]]:
            log.info(f"Processing step {step}...")
            data = getattr(preprocess, step)(data, **cfg["aug_method"][step])
        _save_preprocessed_data(data, cfg)

    log.info("Embedding started...")
    # embed dataset using an defined embedder
    embedder = getattr(embed, cfg["pipeline"]["embed"])(
        data, **cfg["embed"][cfg["pipeline"]["embed"]]
    )

    _, test_entities, train_target, test_taget = extract_ents(
        data
    )  # extract necessary fields from data

    log.info("fit_transform")
    # fit and save embedder
    _, train_embeddings, test_embeddings = embedder.fit_transform()

    # Check if approach allready embedded with configs.
    # Used to achieve multiple embedding iterations without overwriting results
    version = 0
    embeddings_base_path = f'{cfg["file_paths"]["embedded"]}/{data.name}${cfg["pipeline"]["embed"]}$'
    while os.path.exists(f"{embeddings_base_path}train${str(version)}.csv"):
        version += 1

    np.savetxt(
        f"{embeddings_base_path}train${str(version)}.csv",
        train_embeddings,
        delimiter=",",
        fmt="%s",
    )
    np.savetxt(
        f"{embeddings_base_path}test${str(version)}.csv",
        test_embeddings,
        delimiter=",",
        fmt="%s",
    )

    log.info("Classifier fitting started...")
    # fit classifiers given in config file.
    models = {}
    for m in cfg["pipeline"]["evaluate"]:
        log.info(f"fitting {m}...")
        model = getattr(evaluate, m)(**cfg["evaluate"][m])
        model.fit(train_embeddings, train_target)
        models[m] = model

    log.info("Evaluation started...")
    # predict test data and store results into "../data/predicted"
    for m, model in models.items():
        log.info(f"evaluating model {m}")
        predictions = model.predict(test_embeddings)
        version = 0
        predictions_base_path = f'{cfg["file_paths"]["predicted"]}/{data.name}${cfg["pipeline"]["embed"]}${m}'
        while os.path.exists(f"{predictions_base_path}${str(version)}.csv"):
            version += 1
        np.savetxt(
            f"{predictions_base_path}${str(version)}.csv",
            [predictions, test_taget],
            delimiter=",",
            fmt="%s",
        )
        log.info(
            f"Predicted {len(test_entities)} entities with an accuracy of "
            + f"{accuracy_score(test_taget, predictions) * 100 :.4f}%"
        )
        log.info(
            f'resulted in following f scores: micro {f1_score(test_taget, predictions, average="micro")} macro {f1_score(test_taget, predictions, average="macro")}'
        )
        log.info("Confusion Matrix :")
        log.info(confusion_matrix(test_taget, predictions))

    log.info("Save Data...")


if __name__ == "__main__":
    evaluate_one()
