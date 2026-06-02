# This source code is from MKGA (w/ slight adaptations)
#   (https://gitlab.com/patryk.preisner/mkga/-/blob/e5a065403371403bec56686f9425d55a59f4cb1f/src/utils/types.py)
# Copyright (c) 2023 Patryk Preisner
# This source code is licensed under the Apache license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.

from torch import Tensor
from typing import List, Dict, Tuple
import kgbench as kg

# from kgbench.load import fastload, getfile
URI_PREFIX = "http://multimodal-knowledge-graph-augmentation.com/"

ALL_TYPES = [
    "@es",
    "@fy",
    "@nl",
    "@nl-nl",
    "@pt",
    "@ru",
    "blank_node",
    "http://kgbench.info/dt#base64Image",
    "http://www.opengis.net/ont/geosparql#wktLiteral",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString",
    "http://www.w3.org/2001/XMLSchema#anyURI",
    "http://www.w3.org/2001/XMLSchema#boolean",
    "http://www.w3.org/2001/XMLSchema#date",
    "http://www.w3.org/2001/XMLSchema#dateTime",
    "http://www.w3.org/2001/XMLSchema#decimal",
    "http://www.w3.org/2001/XMLSchema#gYear",
    "http://www.w3.org/2001/XMLSchema#nonNegativeInteger",
    "http://www.w3.org/2001/XMLSchema#positiveInteger",
    "http://www.w3.org/2001/XMLSchema#string",
    "iri",
    "none",
]

RDF_ENTITY_TYPES = ["iri", "none", "blank_node"]

RDF_NUMBER_TYPES = [
    "http://www.w3.org/2001/XMLSchema#decimal",
    "http://www.w3.org/2001/XMLSchema#gYear",
    "http://www.w3.org/2001/XMLSchema#nonNegativeInteger",
    "http://www.w3.org/2001/XMLSchema#positiveInteger",
    "http://www.w3.org/2001/XMLSchema#float",
    "http://www.w3.org/2001/XMLSchema#int",
    "http://www.w3.org/2001/XMLSchema#integer",
]

RDF_DECIMAL_TYPES = ["http://www.w3.org/2001/XMLSchema#decimal"]

RDF_DATE_TYPES = [
    "http://www.w3.org/2001/XMLSchema#date",
    "http://www.w3.org/2001/XMLSchema#dateTime",
]

IMAGE_TYPES = ["http://kgbench.info/dt#base64Image"]

GEO_TYPES = ["http://www.opengis.net/ont/geosparql#wktLiteral"]

NONE_TYPES = ["none"]

ALL_LITERALS = [
    "@es",
    "@fy",
    "@nl",
    "@nl-nl",
    "@pt",
    "@ru",
    "http://kgbench.info/dt#base64Image",
    "http://www.opengis.net/ont/geosparql#wktLiteral",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString",
    "http://www.w3.org/2001/XMLSchema#anyURI",
    "http://www.w3.org/2001/XMLSchema#boolean",
    "http://www.w3.org/2001/XMLSchema#date",
    "http://www.w3.org/2001/XMLSchema#dateTime",
    "http://www.w3.org/2001/XMLSchema#decimal",
    "http://www.w3.org/2001/XMLSchema#gYear",
    "http://www.w3.org/2001/XMLSchema#nonNegativeInteger",
    "http://www.w3.org/2001/XMLSchema#positiveInteger",
    "http://www.w3.org/2001/XMLSchema#string",
]

ALL_BUT_NUMBER = [
    "@es",
    "@fy",
    "@nl",
    "@nl-nl",
    "@pt",
    "@ru",
    "http://kgbench.info/dt#base64Image",
    "http://www.opengis.net/ont/geosparql#wktLiteral",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString",
    "http://www.w3.org/2001/XMLSchema#anyURI",
    "http://www.w3.org/2001/XMLSchema#boolean",
    "http://www.w3.org/2001/XMLSchema#date",
    "http://www.w3.org/2001/XMLSchema#dateTime",
    "http://www.w3.org/2001/XMLSchema#string",
]

POTENTIAL_TEXT_TYPES = [
    "@es",
    "@fy",
    "@nl",
    "@nl-nl",
    "@pt",
    "@ru",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString",
    "http://www.w3.org/2001/XMLSchema#string",
        "none",
]


class Data(kg.Data):
    """
    Data Class.

    This class extends the kg.Data class and represents a specific dataset with the following attributes:

    Attributes:
        withheld (Tensor): The withheld data tensor.
        training (Tensor): The training data tensor.
        triples (Tensor): The tensor of triples.
        i2r (List[str]): The list of relation names.
        r2i (Dict[str, int]): The dictionary mapping relation names to their indices.
        i2e (List[Tuple[str, str]]): The list of entity tuples.
        e2i (Dict[Tuple[str, str], int]): The dictionary mapping entity tuples to their indices.
        name (str): The name of the dataset.
        num_entities (int): The number of entities in the dataset.
        num_relations (int): The number of relations in the dataset.
        num_classes (int): The number of classes in the dataset.

    """

    withheld: Tensor
    training: Tensor
    triples: Tensor
    i2r: List[str]
    r2i: Dict[str, int]
    i2e: List[Tuple[str, str]]
    e2i: Dict[Tuple[str, str], int]
    name: str
    num_entities: int
    num_relations: int
    num_classes: int
