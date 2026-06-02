from pykeen import models
from pykeen.training import SLCWATrainingLoop
from pykeen.triples import TriplesFactory
from pykeen.utils import resolve_device
from pyrdf2vec.typings import Embeddings
from torch.optim import Adam
import pandas as pd
from typing import Tuple
import os
import torch
from utils import Data

FILEPATH = "data/preprocessed"


def _data_to_pykeen(data: Data) -> TriplesFactory:
    """
    Convert data to PyKEEN TriplesFactory.

    Args:
        data: The input data to be converted.

    Returns:
        TriplesFactory: The converted PyKEEN TriplesFactory object.

    """
    if not os.path.exists(f"{FILEPATH}/{data.name}.tsv.gz"):
        print("pykeen file does not exist. Writing pykeen file...")
        print(data.name)
        df = pd.DataFrame(data.triples, columns=["h", "r", "t"])  # type: ignore
        df["h"] = df["h"].apply(lambda h: data.i2e[h][0])
        df["r"] = df["r"].apply(lambda r: data.i2r[r])
        df["t"] = df["t"].apply(lambda t: data.i2e[t][0])
        df.to_csv(
            f"{FILEPATH}/{data.name}.tsv.gz",
            index=False,
            sep="\t",
            compression="gzip",
        )
    return TriplesFactory.from_path(f"{FILEPATH}/{data.name}.tsv.gz")


class PykeenEmbedder:
    def __init__(self, embedder, data: Data, optimizer, train_loop_args):
        """
        PyKEEN Embedder Class.

        This class initializes a PyKEEN Embedder object, which encapsulates the embedding model,
        the provided data, optimizer, and other necessary components for training.

        Args:
            embedder: The PyKEEN embedding model.
            data (Data): The input data for embedder.
            optimizer: The optimizer used for training.
            train_loop_args: Additional arguments for the training loop.

        """
        self.data: Data = data
        torch.cuda.empty_cache()
        self.training_triples_factory = _data_to_pykeen(data)
        self.model = embedder(
            triples_factory=self.training_triples_factory, random_seed=42
        )
        self.model = self.model.to(resolve_device("gpu"))
        optimizer = Adam(params=self.model.get_grad_params())
        self.training_loop = SLCWATrainingLoop(
            model=self.model,
            triples_factory=self.training_triples_factory,
            optimizer=optimizer,
        )
        self.train_loop_args = train_loop_args

    def fit_transform(self) -> Tuple[Embeddings, Embeddings, Embeddings]:
        """
        Fit the model and transform the data.

        This method fits a PyKEEN embedding model to the provided data and performs the transformation.

        Returns:
            Tuple[Embeddings, Embeddings, Embeddings]: A tuple containing base train and test embeddings
        """
        self.training_loop.train(
            triples_factory=self.training_triples_factory,
            **self.train_loop_args,
        )
        embeddings = (
            self.model.entity_representations[0]().detach().cpu().numpy()
        )
        reorder = []
        for e in self.data.i2e:
            reorder.append(self.training_triples_factory.entity_to_id[e[0]])
        embeddings = embeddings[reorder]
        # train_entities, test_entities, train_target, test_taget = extract_ents(
        # self.data)  # extract necessary fields from data
        train_embeddings = embeddings[self.data.training[:, 0]]
        test_embeddings = embeddings[self.data.withheld[:, 0]]
        return embeddings, train_embeddings, test_embeddings


class TransE(PykeenEmbedder):
    def __init__(
        self,
        data: Data,
        optimizer="Adam",
        train_loop_args={"num_epochs": 5, "batch_size": 256},
    ):
        """
        TransE Embedder Class.

        This class initializes TransE, being a PyKEEN Embedder object

        Args:
            data (Data): The input data for embedder.
            optimizer: The optimizer used for training.
            train_loop_args: Additional arguments for the training loop.

        """
        super().__init__(models.TransE, data, optimizer, train_loop_args)


class ComplEx(PykeenEmbedder):
    def __init__(
        self,
        data: Data,
        optimizer="Adam",
        train_loop_args={"num_epochs": 5, "batch_size": 256},
    ):
        """
        ComplEx Embedder Class.

        This class initializes TransE, being a PyKEEN Embedder object

        Args:
            data (Data): The input data for embedder.
            optimizer: The optimizer used for training.
            train_loop_args: Additional arguments for the training loop.

        """
        super().__init__(models.ComplEx, data, optimizer, train_loop_args)


class SimplE(PykeenEmbedder):
    def __init__(
        self,
        data: Data,
        optimizer="Adam",
        train_loop_args={"num_epochs": 5, "batch_size": 256},
    ):
        """
        SimplE Embedder Class.

        This class initializes TransE, being a PyKEEN Embedder object

        Args:
            data (Data): The input data for embedder.
            optimizer: The optimizer used for training.
            train_loop_args: Additional arguments for the training loop.

        """
        super().__init__(models.SimplE, data, optimizer, train_loop_args)


class DistMult(PykeenEmbedder):
    def __init__(
        self,
        data: Data,
        optimizer="Adam",
        train_loop_args={"num_epochs": 5, "batch_size": 256},
    ):
        """
        DistMult Embedder Class.

        This class initializes TransE, being a PyKEEN Embedder object

        Args:
            data (Data): The input data for embedder.
            optimizer: The optimizer used for training.
            train_loop_args: Additional arguments for the training loop.

        """
        super().__init__(models.DistMult, data, optimizer, train_loop_args)
