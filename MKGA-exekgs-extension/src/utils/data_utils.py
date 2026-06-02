# This source code is from MKGA (w/ slight adaptations)
#   (https://gitlab.com/patryk.preisner/mkga/-/blob/e5a065403371403bec56686f9425d55a59f4cb1f/src/utils/data_utils.py)
# Copyright (c) 2023 Patryk Preisner
# This source code is licensed under the Apache license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.

# from pyrdf2vec.graphs import KG
# from pyrdf2vec.graphs import KG, Vertex

# from kgbench.load import Data
from utils import Data
from utils import RDF_NUMBER_TYPES
from typing import List, Dict, Tuple
import torch


def update_dataset_name(data: Data, preprocess_args, preprocess_steps) -> Data:
    """
    Update the name attribute of the data object according to preprocess steps applied.

    Args:
        data (Data): The Data object representing the dataset.
        preprocess_args: The arguments for the data preprocessing steps.
        preprocess_steps: The list of preprocessing steps to be applied.

    Returns:
        Data: The updated Data object with augmented name reference.

    """
    for i in range(len(preprocess_steps)):
        if i < len(preprocess_steps):
            data.name = data.name + "+"
        data.name = data.name + f"{preprocess_steps[i]}"
        keys = list(preprocess_args[preprocess_steps[i]].keys())

        for j in range(len(keys)):
            if j < len(keys):
                data.name = data.name + "-"
            data.name = (
                data.name
                + f"{keys[j]}@{str(preprocess_args[preprocess_steps[i]][keys[j]])}"
            )
    return data


def ensure_data_symmetry(data: Data) -> Data:
    """
    Ensure symmetry in the dataset by adding reverse relations. a <-> b

    This method takes a Data object representing a dataset and ensures symmetry by adding reverse relations
    for any missing or duplicated symmetric relations.

    Args:
        data (Data): The Data object representing the dataset.

    Returns:
        Data: The updated Data object with symmetry ensured.

    """
    for t in data.triples:
        t[0] = torch.tensor(data.e2i[data.i2e[t[0]]], dtype=torch.int32)
        t[1] = torch.tensor(data.r2i[data.i2r[t[1]]], dtype=torch.int32)
        t[2] = torch.tensor(data.e2i[data.i2e[t[2]]], dtype=torch.int32)

    for t in data.training:
        t[0] = torch.tensor(data.e2i[data.i2e[t[0]]])

    for t in data.withheld:
        t[0] = torch.tensor(data.e2i[data.i2e[t[0]]])
    base_e_unique = torch.unique(
        torch.cat([data.triples[:, 0], data.triples[:, 2]])
    )
    base_r_unique = torch.unique(data.triples[:, 1])

    new_e2i = {}
    new_i2e = []

    for i in range(len(data.i2e)):
        if i in base_e_unique.numpy():
            new_e2i[data.i2e[i]] = len(new_i2e)
            new_i2e.append(data.i2e[i])

    new_r2i = {}
    new_i2r = []

    for i in range(len(data.i2r)):
        if i in base_r_unique.numpy():
            new_r2i[data.i2r[i]] = len(new_i2r)
            new_i2r.append(data.i2r[i])

    for t in data.triples:
        t[0] = torch.tensor(new_e2i[data.i2e[t[0]]], dtype=torch.int32)
        t[1] = torch.tensor(new_r2i[data.i2r[t[1]]], dtype=torch.int32)
        t[2] = torch.tensor(new_e2i[data.i2e[t[2]]], dtype=torch.int32)

    for t in data.training:
        t[0] = torch.tensor(new_e2i[data.i2e[t[0]]])

    for t in data.withheld:
        t[0] = torch.tensor(new_e2i[data.i2e[t[0]]])

    # update metedata
    data.num_entities = len(new_i2e)
    data.num_relations = len(new_i2r)

    # update data
    data.i2e = new_i2e
    data.e2i = new_e2i
    data.i2r = new_i2r
    data.r2i = new_r2i
    return data


def extract_ents(
    data: Data,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Extract train and test entities from the dataset.

    This method takes a Data object representing a dataset and extracts the train and test entities from the dataset.

    Args:
        data (Data): The Data object representing the dataset.

    Returns:
        Tuple[List[str],List[str],List[str],List[str]]: tuple with entity lists:
        train_entities, test_entities, train_target, test_taget

    """
    train_entities = []
    train_target = []
    for d in data.training:
        ent = data.i2e[d[0]][0]
        train_entities.append(ent)
        train_target.append(int(d[1]))

    test_entities = []
    test_taget = []
    for d in data.withheld:
        ent = data.i2e[d[0]][0]
        test_entities.append(ent)
        test_taget.append(int(d[1]))

    return train_entities, test_entities, train_target, test_taget


# def data_to_kg(data: Data) -> KG:
#     """
#     Convert KG using kgbench Data object representation into pyrdf2vec KG representation.

#     This method takes a kgbench Data object representing a dataset and converts it into a pyrdf2vec KG representation.

#     Args:
#         data (Data): The kgbench Data object representing the dataset.

#     Returns:
#         KnowledgeGraph: The pyrdf2vec KG representation of the dataset.

#     """
#     kg = KG()
#     for triple in data.triples:
#         subj = Vertex(*[data.i2e[triple[0]][0]])
#         obj = Vertex(*[data.i2e[triple[2]][0]])
#         pred = Vertex(
#             *[data.i2r[triple[1]]],
#             **{"predicate": True, "vprev": subj, "vnext": obj},
#         )
#         kg.add_walk(subj, pred, obj)
#     return kg


def get_p_types(data: Data) -> Dict[str, Tuple[List[str], List[str]]]:
    """
    Get property types from the dataset.

    This method takes a Data object representing a dataset and extracts the property types from the dataset.

    Args:
        data (Data): The Data object representing the dataset.

    Returns:
        Dict[str, Tuple[List[str], List[str]]]: A dictionary mapping property names to their corresponding
                                                domain and range types.

    """
    p_types = {}
    for triple in data.triples:
        o_type = data.i2e[triple[0]][1]
        s_type = data.i2e[triple[2]][1]
        p = data.i2r[triple[1]]

        if p not in p_types:
            p_types[p] = ([o_type], [s_type])
        else:
            if o_type not in p_types[p][0]:
                p_types[p][0].append(o_type)
            if s_type not in p_types[p][1]:
                p_types[p][1].append(s_type)
    return p_types


def get_relevant_relations(data: Data, relevant_types: List[str]) -> List[int]:
    """
    Get relevant relations from the dataset based on provided predicate types.

    This method takes a Data object representing a KG and a list of relevant datatypes, and returns a list
    of indices corresponding to the relevant relations in the dataset.

    Args:
        data (Data): The Data object representing the dataset.
        relevant_types (List[str]): The list of relevant types for filtering the relations.

    Returns:
        List[int]: A list of indices corresponding to the relevant relations in the dataset.

    """

    relevant_types_has_boolean_or_string = (
        "http://www.w3.org/2001/XMLSchema#boolean" in relevant_types
        or "http://www.w3.org/2001/XMLSchema#string" in relevant_types
    )

    p_types = get_p_types(data)
    relevent_relations: List[int] = []
    for ptk, ptv in p_types.items():
        for nt in relevant_types:
            if nt in ptv[1] and data.r2i[ptk] not in relevent_relations:
                if not relevant_types_has_boolean_or_string and (
                    "http://www.w3.org/2001/XMLSchema#boolean" in ptv[1]
                ):
                    continue
                relevent_relations.append(data.r2i[ptk])
    return relevent_relations


def add_triple(
    data: Data, s: Tuple[str, str], p: str, o: Tuple[str, str], verbose=0
) -> Data:
    """
    Add a triple to the kgbench Data object.

    This method takes a Data object representing a dataset, along with subject (s), predicate (p), and object (o)
    representing the components of a triple, and adds the triple to the dataset.

    NOTE: as this method adds one triple at a time, vectorized extensions to triples are prefered whenever applicable

    Args:
        data (Data): The Data object representing the dataset.
        s (Tuple[str, str]): The subject of the triple as a tuple of entity type and entity value.
        p (str): The predicate of the triple.
        o (Tuple[str, str]): The object of the triple as a tuple of entity type and entity value.
        verbose (int, optional): Verbosity level for printing progress or debug information. Defaults to 0.

    Returns:
        Data: The updated Data object with the added triple.

    """
    if s not in data.i2e:
        new_id = len(data.i2e)
        data.e2i[s] = new_id
        data.i2e.append(s)
        data.num_entities += 1
        if verbose > 0:
            print(f"created new entity:")
            print(f"{data.e2i[s]} - {s}")
    if o not in data.i2e:
        new_id = len(data.i2e)
        data.e2i[o] = new_id
        data.i2e.append(o)
        data.num_entities += 1
        if verbose > 0:
            print(f"created new entity:")
            print(f"{data.e2i[o]} - {o}")

    if p not in data.i2r:
        new_id = len(data.i2r)
        data.r2i[p] = new_id
        data.i2r.append(p)
        data.num_relations += 1
        if verbose > 0:
            print(f"created new relation:")
            print(f"{data.r2i[p]} - {p}")
    si = data.e2i[s]
    pi = data.r2i[p]
    oi = data.e2i[o]
    if verbose > 1:
        print(f"added triple:")
        print(f"{si} - {pi} - {oi}")
    new_triple = torch.tensor([[si, pi, oi]], dtype=torch.int32)
    data.triples = torch.cat((data.triples, new_triple), 0)
    return data


def delete_r(data: Data, r) -> Data:
    """
    Delete triples with a specific relation from the dataset.

    This method takes a Data object representing a dataset and a relation (r), and deletes all the triples
    with the specified relation from the dataset.

    Args:
        data (Data): The Data object representing the dataset.
        r (str): The relation to be deleted.

    Returns:
        Data: The updated Data object with the specified relation triples deleted.

    """
    # get subset data
    filtered = data.triples[~(torch.isin(data.triples[:, 1], r))]
    # get neg e filter
    base_e_unique = torch.unique(
        torch.cat([data.triples[:, 0], data.triples[:, 2]])
    )
    filtered_e_unique = torch.unique(
        torch.cat([filtered[:, 0], filtered[:, 2]])
    )
    neg_e_filter = base_e_unique[
        ~(torch.isin(base_e_unique, filtered_e_unique))
    ]

    # get neg r filter
    base_r_unique = torch.unique(data.triples[:, 1])
    filtered_r_unique = torch.unique(filtered[:, 1])
    neg_r_filter = base_r_unique[
        ~(torch.isin(base_r_unique, filtered_r_unique))
    ]

    # create new e mapping
    new_e2i = {}
    new_i2e = []

    for i in range(len(data.i2e)):
        if i not in neg_e_filter.numpy():
            new_e2i[data.i2e[i]] = len(new_i2e)
            new_i2e.append(data.i2e[i])

    # create new r mapping
    new_r2i = {}
    new_i2r = []

    for i in range(len(data.i2r)):
        if i not in neg_r_filter.numpy():
            new_r2i[data.i2r[i]] = len(new_i2r)
            new_i2r.append(data.i2r[i])

    # apply new mapping for triples
    for t in filtered:
        t[0] = new_e2i[data.i2e[t[0]]]
        t[1] = new_r2i[data.i2r[t[1]]]
        t[2] = new_e2i[data.i2e[t[2]]]

    # create new train & withheld
    new_train = []
    new_withheld = []

    # calculate new train & withheld
    for ent in data.training:
        new_train.append([new_e2i[data.i2e[ent[0].numpy()]], ent[1]])

    for ent in data.withheld:
        new_withheld.append([new_e2i[data.i2e[ent[0].numpy()]], ent[1]])

    # update metedata
    data.num_entities = len(new_i2e)
    data.num_relations = len(new_i2r)

    # update data
    data.triples = filtered
    data.i2e = new_i2e
    data.e2i = new_e2i
    data.i2r = new_i2r
    data.r2i = new_r2i
    data.training = torch.tensor(new_train)
    data.withheld = torch.tensor(new_withheld)

    return data
