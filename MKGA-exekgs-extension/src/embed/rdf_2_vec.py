import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.graphs import KG
import pandas as pd

import torch


from pyrdf2vec.walkers import RandomWalker
from pyrdf2vec.samplers import PageRankSampler


from typing import Tuple

import pyrdf2vec as r2v
from pyrdf2vec.graphs import KG
from pyrdf2vec.typings import Embeddings
from .bfs_random_walker import BFSRandomWalker

from utils.data_utils import data_to_kg, extract_ents
from utils import Data


class RDF2Vec:
    def __init__(
        self,
        data: Data,
        embedding_name="Word2Vec",
        embedding_args={"workers": 4, "epochs": 40},
        walker_name="BFSRandomWalker",
        walker_args={"max_depth": 2, "with_reverse": True, "random_state": 42},
    ):
        """
        RDF2Vec embedder Class.

        This class initializes an RDF2Vec embedder, which performs RDF to vector embeddings using the specified
        embedding and walker techniques.

        Args:
            data: The input RDF data.
            embedding_name: The name of the embedding technique. Defaults to "Word2Vec".
            embedding_args: Additional arguments for the embedding technique. Defaults to {"workers": 4, "epochs": 40}.
            walker_name: The name of the walker technique. Defaults to "BFSRandomWalker".
            walker_args: Additional arguments for the walker technique. Defaults to {"max_depth": 2, "with_reverse": True, "random_state": 42}.

        """
        self.data = data
        torch.cuda.empty_cache()
        # ignore typings given missing typing definitions
        embedder = getattr(r2v.embedders, embedding_name)(  # type: ignore
            **embedding_args
        )
        if walker_name == "BFSRandomWalker":
            walker = BFSRandomWalker(**walker_args)
        else:
            walker = getattr(r2v.walkers, walker_name)(  # type: ignore
                **walker_args
            )  # type: ignore
        self.transformer: RDF2VecTransformer = RDF2VecTransformer(
            embedder, walkers=[walker], verbose=1  # type: ignore
        )  # type: ignore

    def fit_transform(self) -> Tuple[Embeddings, Embeddings, Embeddings]:
        """
        Fit the model and transform the data.

        This method fits a PyKEEN embedding model to the provided data and performs the transformation.

        Returns:
            Tuple[Embeddings, Embeddings, Embeddings]: A tuple containing base train and test embeddings
        """
        kg = data_to_kg(self.data)
        train_entities, test_entities, _, _ = extract_ents(self.data)
        entities = train_entities + test_entities

        embeddings = self.transformer.fit_transform(kg, entities)[0]
        train_embeddings = embeddings[: len(train_entities)]
        test_embeddings = embeddings[len(train_entities) :]
        return embeddings, train_embeddings, test_embeddings
