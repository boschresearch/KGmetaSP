# This source code is from MKGA (w/ slight adaptations)
#   (https://gitlab.com/patryk.preisner/mkga/-/blob/e5a065403371403bec56686f9425d55a59f4cb1f/src/preprocess/subpopulation.py)
# Copyright (c) 2023 Patryk Preisner
# This source code is licensed under the Apache license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.

import numpy as np
import torch
from sklearn.neighbors import LocalOutlierFactor
from torch import Tensor
from typing import List

from preprocess.binning import encode_number_sublist
from utils import (
    RDF_NUMBER_TYPES,
    get_relevant_relations,
    add_triple,
    Data,
    URI_PREFIX,
)


def adapted_kl_divergence(p_dist, q_dist, num_bins=100):
    """
    Calculate the adapted Kullback-Leibler (KL) divergence between two probability distributions.

    This method takes two probability distributions, p_dist and q_dist, and calculates the adapted Kullback-Leibler (KL)
    divergence between them. The number of bins to be used for the histogram-based approximation can be specified using the
    num_bins parameter. The distribution is adapted to be applicable to the subpopulation case in KGs.

    Args:
        p_dist (numpy.ndarray or torch.Tensor): The probability distribution p.
        q_dist (numpy.ndarray or torch.Tensor): The probability distribution q.
        num_bins (int, optional): The number of bins to be used for the histogram-based approximation. Defaults to 100.

    Returns:
        float: The adapted KL divergence between the two probability distributions.

    """
    p_bin = np.histogram(p_dist, num_bins)[0] + 1
    q_bin = np.histogram(q_dist, num_bins)[0] + 1

    sum_l = 0
    for i in range(num_bins):
        sum_l += np.log(p_bin[i] / q_bin[i]) * p_bin[i]
    return round(len(q_dist) / len(p_dist) * sum_l, 5)


def subpopulation_binning(
    data: Data, num_bins=10, use_lof=False, bound_approach="r", **kwargs
):
    """
    Perform subpopulation binning on the dataset.

    This method takes a Data object representing a KG and performs subpopulation binning on the dataset. Subpopulation
    binning involves dividing the dataset into subpopulations based on certain criteria and assigning bins to each
    subpopulation. The number of bins to be used can be specified using the num_bins parameter.

    Args:
        data (Data): The Data object representing the dataset.
        num_bins (int, optional): The number of bins to be used for subpopulation binning. Defaults to 10.
        use_lof (bool, optional): Whether to use Local Outlier Factor (LOF) for outlier detection. Defaults to False.
        bound_approach (str, optional): The approach to determine the constraining approach that should be applied. Defaults to "r".
        **kwargs: Additional keyword arguments for customization or specific binning options.

    Returns:
        Data: The updated Data object with subpopulation binning applied.

    """
    relevent_relations = get_relevant_relations(data, RDF_NUMBER_TYPES)
    if f"{URI_PREFIX}predicat#prevBin" not in data.r2i:
        p = f"{URI_PREFIX}predicat#prevBin"
        new_id = len(data.i2r)
        data.r2i[p] = new_id
        data.i2r.append(p)
        data.num_relations += 1

        p = f"{URI_PREFIX}predicat#nextBin"
        new_id = len(data.i2r)
        data.r2i[p] = new_id
        data.i2r.append(p)
        data.num_relations += 1

    for relation in relevent_relations:
        children: List[Tensor] = []
        if bound_approach == "rv":
            children: List[Tensor] = r_bound_child_extraction(data, relation)
        else:
            children: List[Tensor] = rv_bound_child_extraction(
                data,
                relation,
                exclude_triples_with_obj_values=[
                    "auto",
                    "scale",
                    "auto_deprecated",
                    "sqrt",
                    "warn",
                    "constant",
                    "optimal",
                    "invscaling",
                    "adaptive",
                    "[100]",
                    "(100,)",
                    "(100, 100)",
                    "[10, 10]",
                    "{'name': 'sklearn.model_selection._split.StratifiedKFold', 'parameters': {'n_splits': '2', 'random_state': '54293', 'shuffle': 'true'}}",
                    "{'name': 'sklearn.model_selection._split.StratifiedKFold', 'parameters': {'n_splits': '2', 'random_state': '62501', 'shuffle': 'true'}}",
                    "[100, 100, 100]",
                    "(100, 100, 100)",
                    # below exist only when mlseakg is used
                    "numeric",
                    "float64",
                    "category",
                    "nominal",
                    "int64",
                ],
            )

        for i in range(len(children)):
            augmented_df = data.triples.clone()
            augmented_df = augmented_df[
                (augmented_df[:, 1] == relation)
                & (torch.isin(augmented_df[:, 0], children[i]))
            ]
            sub_df = encode_number_sublist(augmented_df, data.i2e)

            p = f"{URI_PREFIX}predicat#binning{relation}-{i}"
            new_id = len(data.i2r)
            data.r2i[p] = new_id
            data.i2r.append(p)
            data.num_relations += 1

            for b in range(num_bins):
                o = (
                    f"{URI_PREFIX}entity#bin{b+1}-relation{relation}-child{i}",
                    f"{URI_PREFIX}datatype#bin",
                )
                new_id = len(data.i2e)
                data.e2i[o] = new_id
                data.i2e.append(o)
                data.num_entities += 1
                if b > 0:
                    po = (
                        f"{URI_PREFIX}entity#bin{b}-relation{relation}-child{i}",
                        f"{URI_PREFIX}datatype#bin",
                    )
                    data = add_triple(
                        data, o, f"{URI_PREFIX}predicat#prevBin", po
                    )
                    data = add_triple(
                        data, po, f"{URI_PREFIX}predicat#nextBin", o
                    )

            if use_lof:
                lof = LocalOutlierFactor(n_neighbors=200)
                lof.fit(sub_df[:, 1].to(torch.int).reshape(-1, 1))  # type: ignore
                outlier_scores = lof.negative_outlier_factor_
                # Create a new column in the numpy array to store the outlier scores
                threshold = np.percentile(outlier_scores, 5)
                # use the outlier scores to filter out the outliers from the numpy array
                outliers = sub_df[
                    (outlier_scores <= threshold) & (outlier_scores < -1)
                ]
                sub_df = sub_df[
                    (outlier_scores > threshold) | (outlier_scores >= -1)
                ]
                outlier_df = augmented_df[
                    (outlier_scores <= threshold) & (outlier_scores < -1)
                ].clone()
                augmented_df = augmented_df[
                    (outlier_scores > threshold) | (outlier_scores >= -1)
                ]
                if len(outliers) > 0:
                    data.i2r.append(
                        f"{URI_PREFIX}predicat#outlier-{relation}-{i}"
                    )
                    data.r2i[
                        f"{URI_PREFIX}predicat#outlier-{relation}-{i}"
                    ] = data.num_relations

                    data.i2e.append(
                        (
                            f"{URI_PREFIX}entitys#outlier-{relation}-{i}",
                            f"{URI_PREFIX}outlier",
                        )
                    )
                    data.e2i[
                        (
                            f"{URI_PREFIX}entitys#outlier-{relation}-{i}",
                            f"{URI_PREFIX}outlier",
                        )
                    ] = data.num_entities

                    data.num_relations += 1
                    data.num_entities += 1

                    object_mapping = np.vectorize(
                        lambda t: data.e2i[
                            (
                                f"{URI_PREFIX}entitys#outlier-{relation}-{i}",
                                f"{URI_PREFIX}outlier",
                            )
                        ]
                    )

                    predicat_mapping = np.vectorize(
                        lambda t: data.r2i[
                            f"{URI_PREFIX}predicat#outlier-{relation}-{i}"
                        ]
                    )

                    outlier_df[:, 1] = torch.tensor(
                        np.array([predicat_mapping(outliers[:, 0])]),
                        dtype=torch.int32,
                    )
                    outlier_df[:, 2] = torch.tensor(
                        np.array([object_mapping(outliers[:, 0])]),
                        dtype=torch.int32,
                    )
                    data.triples = torch.cat((data.triples, outlier_df), 0)

            sub_df = torch.cat(  # put bins and sub_df together
                (
                    sub_df,
                    torch.from_numpy(  # get numpy solutions back
                        np.digitize(  # assign for each value in sub_df the corresponding bin
                            sub_df[:, 1],
                            np.histogram(  # calculate n bins based on values in sub_df
                                sub_df[:, 1], num_bins
                            )[
                                1
                            ][
                                :-1
                            ],
                        )
                    ).reshape(
                        -1, 1
                    ),  # transfrom x tensor into (x,1) tensor to fit (x,2) shape of sub_df
                ),
                1,
            )

            object_mapping = np.vectorize(
                lambda t: data.e2i[
                    (
                        f"{URI_PREFIX}entity#bin{int(t)}-relation{relation}-child{i}",
                        f"{URI_PREFIX}datatype#bin",
                    )
                ]
            )

            predicat_mapping = np.vectorize(
                lambda t: data.r2i[
                    f"{URI_PREFIX}predicat#binning{relation}-{i}"
                ]
            )

            augmented_df[:, 1] = torch.tensor(
                np.array([predicat_mapping(sub_df[:, 2])]), dtype=torch.int32
            )
            augmented_df[:, 2] = torch.tensor(
                np.array([object_mapping(sub_df[:, 2])]), dtype=torch.int32
            )
            data.triples = torch.cat((data.triples, augmented_df), 0)
    return data


def r_bound_child_extraction(data: Data, relation: int) -> List[Tensor]:
    """
    Extract child entities based on the specified relation from the dataset.

    This method takes a Data object representing a KG and a relation identifier, and extracts the child entities
    associated with the specified relation from the KG. The extracted child entities are returned as a list of tensors.

    Args:
        data (Data): The Data object representing the KG.
        relation (int): The identifier of the relation for which to extract child entities.

    Returns:
        List[Tensor]: The list of tensors representing the extracted child entities.

    """
    triples = data.triples[data.triples[:, 1] == relation]
    parent = triples[:, 0]

    sub_df = encode_number_sublist(triples, data.i2e)

    all_rels = data.triples[torch.isin(data.triples[:, 0], triples[:, 0])]
    r, counts = torch.unique(all_rels[:, 1], return_counts=True)

    r_cnt = torch.cat((r.view(-1, 1), counts.view(-1, 1)), dim=1)
    r_cnt = r_cnt[r_cnt[:, 0] != relation]
    _, indices = torch.sort(r_cnt[:, 1], descending=True)
    r_cnt = r_cnt[indices]

    parent_reminder = parent.clone()
    children: List[Tensor] = []

    for rels in r_cnt:
        pot_child = all_rels[
            (all_rels[:, 1] == rels[0])
            & (torch.isin(all_rels[:, 0], parent_reminder))
        ][:, 0]

        p_dist = sub_df[torch.isin(sub_df[:, 0], parent_reminder)][:, 1]
        q_dist = sub_df[torch.isin(sub_df[:, 0], pot_child)][:, 1]
        kl_div = adapted_kl_divergence(p_dist, q_dist)

        proportion = len(pot_child) / len(parent_reminder)

        print(f"j: {rels[0]} - prop: {proportion} - kl_div: {kl_div}")

        if proportion > 0.98 or proportion < 0.02:
            pass

        # only add if kl >= 500
        elif kl_div < 500:
            pass
        else:
            children.append(pot_child)
            parent_reminder = parent_reminder[
                ~torch.isin(parent_reminder, pot_child)
            ]

    children.append(parent_reminder)
    return children


def rv_bound_child_extraction(
    data: Data, relation: int, exclude_triples_with_obj_values=None
) -> List[Tensor]:
    """
    Extract child entities based on the specified relation-value pair from the dataset.

    This method takes a Data object representing a KG and a relation identifier, and extracts the child entities
    associated with the specified relation bound by values from the KG.
    The extracted child entities are returned as a list of tensors.

    Args:
        data (Data): The Data object representing the KG.
        relation (int): The identifier of the relation for which to extract child entities.

    Returns:
        List[Tensor]: The list of tensors representing the extracted child entities.

    """
    triples = data.triples[data.triples[:, 1] == relation]

    if exclude_triples_with_obj_values:
        exclude_list = []
        for obj in exclude_triples_with_obj_values:
            if (obj, "http://www.w3.org/2001/XMLSchema#string") in data.e2i:
                exclude_list.append(
                    data.e2i[(obj, "http://www.w3.org/2001/XMLSchema#string")]
                )
            elif (
                obj,
                "http://www.w3.org/2000/01/rdf-schema#Literal",
            ) in data.e2i:
                exclude_list.append(
                    data.e2i[
                        (obj, "http://www.w3.org/2000/01/rdf-schema#Literal")
                    ]
                )
            else:
                print(f"Object {obj} not found in data.e2i. Skipping ...")

        exclude_tensor = torch.tensor(exclude_list)
        triples = triples[~torch.isin(triples[:, 2], exclude_tensor)]

    # if all triples have the same object, we can't split them
    if len(torch.unique(triples[:, 2])) == 1:
        print(
            f"All triples have the same object {triples[0, 2]} for relation {data.i2r[relation]}. Skipping relation ..."
        )
        return []

    parent = triples[:, 0]

    sub_df = encode_number_sublist(triples, data.i2e)
    all_rels = data.triples[torch.isin(data.triples[:, 0], triples[:, 0])]
    r, counts = torch.unique(all_rels[:, 1], return_counts=True)

    r_cnt = torch.cat((r.view(-1, 1), counts.view(-1, 1)), dim=1)
    r_cnt = r_cnt[r_cnt[:, 0] != relation]
    _, indices = torch.sort(r_cnt[:, 1], descending=True)
    r_cnt = r_cnt[indices]
    r_cnt = r_cnt[r_cnt[:, 0] != relation]

    parent_reminder = parent.clone()
    children: List[Tensor] = []

    for rels in r_cnt:
        print(f"processing {rels}")
        rv, counts = torch.unique(
            all_rels[all_rels[:, 1] == rels[0]][:, 2], return_counts=True
        )
        rv_cnt = torch.cat((rv.view(-1, 1), counts.view(-1, 1)), dim=1)
        _, indices = torch.sort(rv_cnt[:, 1], descending=True)
        rv_cnt = rv_cnt[indices]

        max_prop = rv_cnt[0, 1] / len(parent_reminder)
        min_prop = rv_cnt[-1, 1] / len(parent_reminder)

        if max_prop < 0.001 or min_prop > 0.999:
            pass
        else:
            for rel_val in rv_cnt:
                if rel_val[1] / len(parent_reminder) < 0.001:
                    break
                pot_child = all_rels[
                    (all_rels[:, 1] == rels[0])
                    & (torch.isin(all_rels[:, 0], parent_reminder))
                    & (all_rels[:, 2] == rel_val[0])
                ][:, 0]
                p_dist = sub_df[torch.isin(sub_df[:, 0], parent_reminder)][
                    :, 1
                ]
                q_dist = sub_df[torch.isin(sub_df[:, 0], pot_child)][:, 1]
                proportion = len(pot_child) / len(parent_reminder)
                kl_div = adapted_kl_divergence(p_dist, q_dist)
                if proportion > 0.1 and proportion < 0.9:
                    print(
                        f"j: {rels[0]} - prop: {proportion} - kl_div: {kl_div}"
                    )
                if proportion > 0.98 or proportion < 0.02:
                    pass
                if kl_div < 500:
                    pass
                else:
                    children.append(pot_child)
                    parent_reminder = parent_reminder[
                        ~torch.isin(parent_reminder, pot_child)
                    ]
        if len(children) > 0:
            break
    children.append(parent_reminder)
    return children


def rel_bound_subpopulation(data: Data, num_bins=10, **kwargs):
    """
    GDA approach, performing subpopulation binning based on relation bounds on the dataset.

    This method takes a Data object representing a KG and performs subpopulation binning based on relation bounds.
    The dataset is divided into subpopulations based on the relations present in the data, and bins are assigned to each
    relevant subpopulation. The number of bins to be used can be specified using the num_bins parameter.

    Args:
        data (Data): The Data object representing the dataset.
        num_bins (int, optional): The number of bins to be used for subpopulation binning. Defaults to 10.
        **kwargs: Additional keyword arguments.

    Returns:
        Data: The updated Data object with subpopulation binning based on relation bounds applied.

    """
    return subpopulation_binning(data, num_bins=num_bins, bound_approach="r")


def rel_val_bound_subpopulation(data: Data, num_bins=10, **kwargs):
    """
    GDA approach, performing subpopulation binning based on relation-value bounds on the dataset.

    This method takes a Data object representing a KG and performs subpopulation binning based on relation-value bounds.
    The dataset is divided into subpopulations based on the relation-value pairs present in the data, and bins are assigned to each
    relevant subpopulation. The number of bins to be used can be specified using the num_bins parameter.

    Args:
        data (Data): The Data object representing the dataset.
        num_bins (int, optional): The number of bins to be used for subpopulation binning. Defaults to 10.
        **kwargs: Additional keyword arguments.

    Returns:
        Data: The updated Data object with subpopulation binning based on relation-value bounds applied.

    """
    return subpopulation_binning(data, num_bins=num_bins, bound_approach="rv")


def rel_bound_LOF_subpopulation(data: Data, num_bins=10, **kwargs):
    """
    GDA approach, performing subpopulation binning based on relation bounds on the dataset with LOF outlier removal.

    This method takes a Data object representing a KG and performs subpopulation binning based on relation bounds.
    The dataset is divided into subpopulations based on the relations present in the data,
    outliers according to LOF are removed from the KG, and bins are assigned to each
    relevant subpopulation. The number of bins to be used can be specified using the num_bins parameter.

    Args:
        data (Data): The Data object representing the dataset.
        num_bins (int, optional): The number of bins to be used for subpopulation binning. Defaults to 10.
        **kwargs: Additional keyword arguments.

    Returns:
        Data: The updated Data object with subpopulation binning based on relation LOF bounds applied.

    """
    return subpopulation_binning(
        data, num_bins=num_bins, use_lof=True, bound_approach="r"
    )


def rel_val_bound_LOF_subpopulation(data: Data, num_bins=10, **kwargs):
    """
    GDA approach, performing subpopulation binning based on relation-value bounds on the dataset with LOF outlier removal.

    This method takes a Data object representing a KG and performs subpopulation binning based on relation-value bounds.
    The dataset is divided into subpopulations based on the relation-value pairs present in the data,
    outliers according to LOF are removed from the KG, and bins are assigned to each
    relevant subpopulation. The number of bins to be used can be specified using the num_bins parameter.

    Args:
        data (Data): The Data object representing the dataset.
        num_bins (int, optional): The number of bins to be used for subpopulation binning. Defaults to 10.
        **kwargs: Additional keyword arguments.

    Returns:
        Data: The updated Data object with subpopulation binning based on relation-value LOF bounds applied.

    """
    return subpopulation_binning(
        data, num_bins=num_bins, use_lof=True, bound_approach="rv"
    )
