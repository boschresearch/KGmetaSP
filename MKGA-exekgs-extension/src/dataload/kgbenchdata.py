# This source code is from MKGA (w/ slight adaptations)
#   (https://gitlab.com/patryk.preisner/mkga/-/blob/e5a065403371403bec56686f9425d55a59f4cb1f/src/dataload/kgbenchdata.py)
# Copyright (c) 2023 Patryk Preisner
# This source code is licensed under the Apache license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.

import pathlib
import site
import kgbench as kg
import os
import pickle
import re

from kgbench.load import fastload, load_indices, load_entities, getfile
import torch

from utils import Data
from utils import (
    get_relevant_relations,
    RDF_DECIMAL_TYPES,
    ensure_data_symmetry,
)

HERE = pathlib.Path(__file__).parent

BASE_FILE_PATH = str(HERE / ".." / ".." / "data" / "raw")


class DataWithoutClassInformation(Data):
    """
    Class representing a dataset.

    """

    def __init__(
        self,
        dir,
        final=False,
        use_torch=False,
        catval=False,
        name="unnamed_dataset",
    ):

        self.name = name

        self.triples = None
        """ The edges of the knowledge graph (the triples), represented by their integer indices. A (m, 3) numpy 
            or pytorch array.
        """

        self.i2r, self.r2i = None, None

        self.i2e = None
        """ A mapping from an integer index to an entity representation. An entity is either a simple string indicating the label 
            of the entity (a url, blank node or literal), or it is a pair indicating the datatype and the label (in that order).
        """

        self.e2i = None
        """ A dictionary providing the inverse mappring of i2e
        """

        self.num_entities = None
        """ Total number of distinct entities (nodes) in the graph """

        self.num_relations = None
        """ Total number of distinct relation types in the graph """

        self.num_classes = None
        """ Total number of classes in the classification task """

        self.training = None
        """ Training data: a matrix with entity indices in column 0 and class indices in column 1.
            In non-final mode, this is the training part of the train/val/test split. In final mode, the training part, 
            possibly concatenated with the validation data.
        """

        self.withheld = None
        """ Validation/testing data: a matrix with entity indices in column 0 and class indices in column 1.
            In non-final mode this is the validation data. In final mode this is the testing data.
        """

        self._dt_l2g = {}
        self._dt_g2l = {}

        self._datatypes = None
        if dir is not None:

            self.torch = use_torch

            self.triples = fastload(getfile(dir, "triples.int.csv.gz"))

            self.i2r, self.r2i = load_indices(
                getfile(dir, "relations.int.csv")
            )
            self.i2e, self.e2i = load_entities(getfile(dir, "nodes.int.csv"))

            self.num_entities = len(self.i2e)
            self.num_relations = len(self.i2r)

            if use_torch:  # this should be constant-time/memory
                self.triples = torch.from_numpy(self.triples)


def amplus(final=True, torch=True, prune_dist=None, **kwargs) -> Data:
    """
    Load the Amplus dataset.

    Args:
        final (bool): Flag indicating whether to use the final version of the dataset. Defaults to True.
        torch (bool): Flag indicating whether to convert the dataset to Torch tensors. Defaults to True.
        prune_dist (float): Distance threshold for pruning. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        Data: The loaded Amplus dataset.

    """
    if _pickle_exist(
        "amplus", final=final, torch=torch, prune_dist=prune_dist
    ):
        data = _load_pickle(
            "amplus", final=final, torch=torch, prune_dist=prune_dist
        )
    else:
        pickle_name = f'amplus_{"final" if final else ""}_{"torch" if torch else ""}_{prune_dist}'
        data = _load("amplus", final=final, torch=torch, prune_dist=prune_dist)
        print("clean up data (transform decimals)")
        if (
            data != None
            and data.triples != None
            and data.i2e != None
            and data.e2i != None
        ):
            # clean up decimal values for further discretization processes, given bad data quality.
            relevant_relations = get_relevant_relations(
                data, RDF_DECIMAL_TYPES
            )
            for relation in relevant_relations:
                for triple in data.triples[data.triples[:, 1] == relation]:
                    regex_search = re.search(
                        r"[0-9]+\.?[0-9]*", data.i2e[triple[2]][0]
                    )
                    new_string = "0"
                    if regex_search != None:
                        new_string = regex_search.group()  # get first result
                    data.e2i.pop(data.i2e[triple[2]])
                    data.i2e[triple[2]] = (new_string, data.i2e[triple[2]][1])
                    data.e2i[data.i2e[triple[2]]] = int(triple[2])
            data = ensure_data_symmetry(data)
        print("save file...")
        with open(f"{BASE_FILE_PATH}/{pickle_name}.pickle", "wb") as f:
            pickle.dump(data, f)
    return data


def dmgfull(final=True, torch=True, prune_dist=None, **kwargs) -> Data:
    """
    Load the dmgfull dataset.

    Args:
        final (bool): Flag indicating whether to use the final version of the dataset. Defaults to True.
        torch (bool): Flag indicating whether to convert the dataset to Torch tensors. Defaults to True.
        prune_dist (float): Distance threshold for pruning. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        Data: The loaded dmgfull dataset.

    """
    if _pickle_exist(
        "dmgfull", final=final, torch=torch, prune_dist=prune_dist
    ):
        data = _load_pickle(
            "dmgfull", final=final, torch=torch, prune_dist=prune_dist
        )
    else:
        pickle_name = f'dmgfull_{"final" if final else ""}_{"torch" if torch else ""}_{prune_dist}'
        data = _load(
            "dmgfull", final=final, torch=torch, prune_dist=prune_dist
        )
        print("save file...")
        with open(f"{BASE_FILE_PATH}/{pickle_name}.pickle", "wb") as f:
            pickle.dump(data, f)
    return data


def dmg777k(final=True, torch=True, prune_dist=None, **kwargs) -> Data:
    """
    Load the dmg777k dataset.

    Args:
        final (bool): Flag indicating whether to use the final version of the dataset. Defaults to True.
        torch (bool): Flag indicating whether to convert the dataset to Torch tensors. Defaults to True.
        prune_dist (float): Distance threshold for pruning. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        Data: The loaded dmg777k dataset.

    """
    if _pickle_exist(
        "dmg777k", final=final, torch=torch, prune_dist=prune_dist
    ):
        data = _load_pickle(
            "dmg777k", final=final, torch=torch, prune_dist=prune_dist
        )
    else:
        pickle_name = f'dmg777k_{"final" if final else ""}_{"torch" if torch else ""}_{prune_dist}'
        data = _load(
            "dmg777k", final=final, torch=torch, prune_dist=prune_dist
        )
        print("save file...")
        with open(f"{BASE_FILE_PATH}/{pickle_name}.pickle", "wb") as f:
            pickle.dump(data, f)
    return data


def mdgenre(final=True, torch=True, prune_dist=None, **kwargs) -> Data:
    """
    Load the mdgenre dataset.

    Args:
        final (bool): Flag indicating whether to use the final version of the dataset. Defaults to True.
        torch (bool): Flag indicating whether to convert the dataset to Torch tensors. Defaults to True.
        prune_dist (float): Distance threshold for pruning. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        Data: The loaded mdgenre dataset.

    """
    if _pickle_exist(
        "mdgenre", final=final, torch=torch, prune_dist=prune_dist
    ):
        data = _load_pickle(
            "mdgenre", final=final, torch=torch, prune_dist=prune_dist
        )
    else:
        pickle_name = f'mdgenre_{"final" if final else ""}_{"torch" if torch else ""}_{prune_dist}'
        data = _load(
            "mdgenre", final=final, torch=torch, prune_dist=prune_dist
        )
        print("save file...")
        with open(f"{BASE_FILE_PATH}/{pickle_name}.pickle", "wb") as f:
            pickle.dump(data, f)
    return data


def exekgs(final=True, torch=True, prune_dist=None, **kwargs) -> Data:
    """
    Load the exekgs dataset.

    Args:
        final (bool): Flag indicating whether to use the final version of the dataset. Defaults to True.
        torch (bool): Flag indicating whether to convert the dataset to Torch tensors. Defaults to True.
        prune_dist (float): Distance threshold for pruning. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        Data: The loaded exekgs dataset.

    """
    if _pickle_exist(
        "exekgs", final=final, torch=torch, prune_dist=prune_dist
    ):
        data = _load_pickle(
            "exekgs", final=final, torch=torch, prune_dist=prune_dist
        )
    else:
        pickle_name = f'exekgs_{"final" if final else ""}_{"torch" if torch else ""}_{prune_dist}'
        data = _load(
            site.getsitepackages()[0] + "/kgbench/datasets/exekgs.tgz",
            final=final,
            torch=torch,
            prune_dist=prune_dist,
        )
        print("save file...")
        with open(f"{BASE_FILE_PATH}/{pickle_name}.pickle", "wb") as f:
            pickle.dump(data, f)
    return data


def exekgs_with_mlseakg(
    final=True, torch=True, prune_dist=None, **kwargs
) -> Data:
    """
    Load the exekgs_with_mlseakg dataset.

    Args:
        final (bool): Flag indicating whether to use the final version of the dataset. Defaults to True.
        torch (bool): Flag indicating whether to convert the dataset to Torch tensors. Defaults to True.
        prune_dist (float): Distance threshold for pruning. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        Data: The loaded exekgs_with_mlseakg dataset.

    """
    if _pickle_exist(
        "exekgs_with_mlseakg", final=final, torch=torch, prune_dist=prune_dist
    ):
        data = _load_pickle(
            "exekgs_with_mlseakg",
            final=final,
            torch=torch,
            prune_dist=prune_dist,
        )
    else:
        pickle_name = f'exekgs_with_mlseakg_{"final" if final else ""}_{"torch" if torch else ""}_{prune_dist}'
        data = _load(
            site.getsitepackages()[0]
            + "/kgbench/datasets/exekgs_with_mlseakg.tgz",
            final=final,
            torch=torch,
            prune_dist=prune_dist,
        )
        print("save file...")
        with open(f"{BASE_FILE_PATH}/{pickle_name}.pickle", "wb") as f:
            pickle.dump(data, f)
    return data


def _load(
    dataset_name="dmg777k", final=True, torch=True, prune_dist=None, **kwargs
) -> Data:
    """
    Base load method,
    used to load any kgbench dataset and transforming into a "not None" variant of the dataset.

    Args:
        dataset_name (str): kgbench dataset name. possible: ['amplus', 'dblp', 'dmgfull', 'dmg777k', 'mdgenre']
        final (str): kgbench dataset name. possible: ['amplus', 'dblp', 'dmgfull', 'dmg777k', 'mdgenre']
    Returns:
        Data: KG, encoded using kgbench Dataset object

    Args:
        dataset_name (str): Name of kgbench dataset name.
        possible: ['amplus', 'dblp', 'dmgfull', 'dmg777k', 'mdgenre']. Defaults to "dmg777k".
        final (bool): Flag indicating whether to load the final version of the dataset. Defaults to True.
        torch (bool): Flag indicating whether to convert the dataset to Torch tensors. Defaults to True.
        prune_dist (float): Distance threshold for pruning. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        Data: The loaded dataset.

    """

    # ignore warning, datatype loaded here is extended by datatypes defined in the framework specific Data type
    # if dataset_name == "exekgs" or dataset_name == "exekgs_with_mlseakg":
    #     data: DataWithoutClassInformation = DataWithoutClassInformation(kg.load(dataset_name, final, torch, prune_dist=prune_dist))  # type: ignore
    # else:
    data: Data = kg.load(dataset_name, final, torch, prune_dist=prune_dist)  # type: ignore

    # fixing https://github.com/pbloem/kgbench-loader/issues/2 for unpruned datasets
    if prune_dist == None:
        clean_e2i = {}
        for e in data.e2i.keys():
            clean_e2i[e[1]] = e[0]
        data.e2i = clean_e2i
    print("ensure data symmetry...")
    data = ensure_data_symmetry(data)
    return data


def _pickle_exist(
    dataset_name="dmg777k", final=True, torch=True, prune_dist=None
) -> bool:
    """
    Check if pickled version of dataset exists.

    Args:
        dataset_name (str): Name of the dataset. Defaults to "dmg777k".
        final (bool): Flag indicating whether to load the final version of the dataset. Defaults to True.
        torch (bool): Flag indicating whether to convert the dataset to Torch tensors. Defaults to True.
        prune_dist (float): Distance threshold for pruning. Defaults to None.
    Returns:
        bool: True if the pickled dataset exists, False otherwise.

    """
    pickle_name = f'{dataset_name}_{"final" if final else ""}_{"torch" if torch else ""}_{prune_dist}'
    return os.path.exists(f"{BASE_FILE_PATH}/{pickle_name}.pickle")


def _load_pickle(
    dataset_name="dmg777k", final=True, torch=True, prune_dist=None
) -> Data:
    """
    Load the dataset from a pickled file.

    Args:
        dataset_name (str): Name of the dataset to load. Defaults to "dmg777k".
        final (bool): Flag indicating whether to load the final version of the dataset. Defaults to True.
        torch (bool): Flag indicating whether to convert the dataset to Torch tensors. Defaults to True.
        prune_dist (float): Distance threshold for pruning. Defaults to None.

    Returns:
        Data: The loaded dataset.

    """
    pickle_name = f'{dataset_name}_{"final" if final else ""}_{"torch" if torch else ""}_{prune_dist}'
    with open(f"{BASE_FILE_PATH}/{pickle_name}.pickle", "rb") as f:
        data: Data = pickle.load(f)
    return data
