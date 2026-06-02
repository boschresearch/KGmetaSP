# This source code is from MKGA (w/ slight adaptations)
#   (https://gitlab.com/patryk.preisner/mkga/-/blob/e5a065403371403bec56686f9425d55a59f4cb1f/src/preprocess/binning.py)
# Copyright (c) 2023 Patryk Preisner
# This source code is licensed under the Apache license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.

import pandas as pd
import math
import numpy as np

import torch

from utils import (
    RDF_NUMBER_TYPES,
    get_relevant_relations,
    add_triple,
    ALL_LITERALS,
)

from utils import Data
from typing import List, Tuple


import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from utils import URI_PREFIX
import numpy as np
from sklearn.neighbors import LocalOutlierFactor


def simplistic_approach(data: Data, **kwargs) -> Data:
    """
    Apply a simplistic approach to the dataset.

    This method applies a simplistic approach to the given Data object representing a dataset.
    Here no literals are altered within the KG.
    This method is a boilerplate, implemented to fit into the general GDA approach structure.

    Args:
        data (Data): The Data object representing the dataset.
        **kwargs: Additional keyword arguments.

    Returns:
        Data: The updated Data object after applying the simplistic approach.

    """
    return data


def encode_number_sublist(
    triples: torch.Tensor, i2e: List[Tuple[str, str]]
) -> torch.Tensor:
    """
    Encode number sublist of a given tensor using the provided index-to-entity mapping.

    This method takes a tensor (triples) representing a the triples of a KG and a list of index-to-entity mapping (i2e),
    and encodes the number sublist in the tensor using the corresponding index values from the mapping.

    Args:
        triples (torch.Tensor): The input tensor representing KG triples
        i2e (List[Tuple[str, str]]): The entity-to-index mapping.

    Returns:
        torch.Tensor: The updated tensor with the number sublist encoded.

    """
    sub_triples = triples.clone()
    sub_triples = sub_triples.to(torch.float32)
    for i in range(len(triples)):
        # try:
        sub_triples[i, 1] = torch.tensor(
            float(i2e[triples[i, 2]][0]), dtype=torch.float32
        )
        # except ValueError as e:
        #     sub_triples[i, 1] = torch.tensor(
        #         float(0), dtype=torch.float32
        #     )
        #     print("\t\t\t\t" + str(e))
    sub_triples = sub_triples[:, :2]
    return sub_triples


def bin_numbers(
    data: Data,
    num_bins=3,
    use_lof=False,
    num_bins_as_percent=False,
    **kwargs,
) -> Data:
    """
    Bin numbers in the dataset.

    This method takes a Data object representing a KG and performs binning on numerical values in the KG.
    Various options can be specified using the provided keyword arguments. Binning can be performed based on a fixed
    number of bins (num_bins), using Local Outlier Factor (LOF) for binning (use_lof), treating num_bins as a percentage
    (num_bins_as_percent), or using equal height binning (equal_height_binning).

    Args:
        data (Data): The Data object representing the KG.
        num_bins (int, optional): The number of bins to use for binning. Defaults to 3.
        use_lof (bool, optional): Flag indicating whether to use Local Outlier Factor (LOF) for binning. Defaults to False.
        num_bins_as_percent (bool, optional): Flag indicating whether to treat num_bins as a percentage. Defaults to False.
        equal_height_binning (bool, optional): Flag indicating whether to use equal height binning. Defaults to False.
        **kwargs: Additional keyword arguments for customization or specific binning options.

    Returns:
        Data: The updated KG object with the binned literals augmenting the KG.
    """
    relevent_relations = get_relevant_relations(
        data, relevant_types=RDF_NUMBER_TYPES
    )

    bin_percent = num_bins / 100

    # Add bin specific nodes to KG
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

    for r in relevent_relations:
        p = f"{URI_PREFIX}predicat#binning{r}"
        new_id = len(data.i2r)
        data.r2i[p] = new_id
        data.i2r.append(p)
        data.num_relations += 1

    # Apply binning on a per predicate basis:
    for relation in relevent_relations:
        sub_df = encode_number_sublist(
            data.triples[data.triples[:, 1] == relation], data.i2e
        )

        if num_bins_as_percent:
            num_bins = math.floor(len(sub_df[:, 1].unique()) * bin_percent)
            if num_bins < 1:
                num_bins = 1

        # Add predicate specific nodes to KG
        for b in range(num_bins):
            o = (
                f"{URI_PREFIX}entity#binning{b+1}#relation{relation}",
                f"{URI_PREFIX}datatype#bin",
            )
            new_id = len(data.i2e)
            data.e2i[o] = new_id
            data.i2e.append(o)
            data.num_entities += 1
            if (
                f"{URI_PREFIX}entity#binning{b}#relation{relation}",
                f"{URI_PREFIX}datatype#bin",
            ) in data.e2i:
                data = add_triple(
                    data,
                    o,
                    f"{URI_PREFIX}predicat#prevBin",
                    (
                        f"{URI_PREFIX}entity#binning{b}#relation{relation}",
                        f"{URI_PREFIX}datatype#bin",
                    ),
                )
                data = add_triple(
                    data,
                    (
                        f"{URI_PREFIX}entity#binning{b}#relation{relation}",
                        f"{URI_PREFIX}datatype#bin",
                    ),
                    f"{URI_PREFIX}predicat#nextBin",
                    o,
                )

        augmented_df = data.triples.clone()
        augmented_df = augmented_df[augmented_df[:, 1] == relation]

        # calculate LOF outliers before binning
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
            # Add predicate-outliner specific nodes to KG
            if len(outliers) > 0:
                data.i2r.append(f"{URI_PREFIX}predicat#outlier-{relation}")
                data.r2i[f"{URI_PREFIX}predicat#outlier-{relation}"] = (
                    data.num_relations
                )

                data.i2e.append(
                    (
                        f"{URI_PREFIX}entitys#outlier-{relation}",
                        f"{URI_PREFIX}outlier",
                    )
                )
                data.e2i[
                    (
                        f"{URI_PREFIX}entitys#outlier-{relation}",
                        f"{URI_PREFIX}outlier",
                    )
                ] = data.num_entities

                data.num_relations += 1
                data.num_entities += 1

                object_mapping = np.vectorize(
                    lambda t: data.e2i[
                        (
                            f"{URI_PREFIX}entitys#outlier-{relation}",
                            f"{URI_PREFIX}outlier",
                        )
                    ]
                )

                # add and remap base values
                predicat_mapping = np.vectorize(
                    lambda t: data.r2i[
                        f"{URI_PREFIX}predicat#outlier-{relation}"
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

        # BINNING APPLIED HERE:
        # numpy is used here since torch.histc was not working for some reason.
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
                    f"{URI_PREFIX}entity#binning{int(t)}#relation{relation}",
                    f"{URI_PREFIX}datatype#bin",
                )
            ]
        )

        predicat_mapping = np.vectorize(
            lambda t: data.r2i[f"{URI_PREFIX}predicat#binning{relation}"]
        )

        augmented_df[:, 1] = torch.tensor(
            np.array([predicat_mapping(sub_df[:, 2])]), dtype=torch.int32
        )
        augmented_df[:, 2] = torch.tensor(
            np.array([object_mapping(sub_df[:, 2])]), dtype=torch.int32
        )
        data.triples = torch.cat((data.triples, augmented_df), 0)

    return data


def bin_numbers_5(data: Data, **kwargs) -> Data:
    """
    GDA approach, binning number literals in a KG into 5 bins per predicate.

    This method takes a Data object representing a KG and performs binning on numerical literals in the dataset
    using 5 bins.

    Args:
        data (Data): The Data object representing the dataset.
        **kwargs: Additional keyword arguments.

    Returns:
        Data: The augmented Data object with the number literals binned using 5 bins.

    """
    return bin_numbers(data=data, num_bins=5, use_lof=False)


def bin_numbers_10(data: Data, **kwargs) -> Data:
    """
    GDA approach, binning number literals in a KG into 10 bins per predicate.

    This method takes a Data object representing a KG and performs binning on numerical literals in the dataset
    using 10 bins.

    Args:
        data (Data): The Data object representing the dataset.
        **kwargs: Additional keyword arguments.

    Returns:
        Data: The augmented Data object with the number literals binned using 10 bins.

    """
    return bin_numbers(data=data, num_bins=10, use_lof=False)


def bin_numbers_100(data: Data, **kwargs) -> Data:
    """
    GDA approach, binning number literals in a KG into 100 bins per predicate.

    This method takes a Data object representing a KG and performs binning on numerical literals in the dataset
    using 100 bins.

    Args:
        data (Data): The Data object representing the dataset.
        **kwargs: Additional keyword arguments.

    Returns:
        Data: The augmented Data object with the number literals binned using 100 bins.

    """
    return bin_numbers(data=data, num_bins=100, use_lof=False)


def bin_numbers_lof_5(data: Data, **kwargs) -> Data:
    """
    GDA approach, binning number literals in a KG into 5 bins per predicate. Removing outliers detected with LOF.

    This method takes a Data object representing a KG and performs binning on numerical literals in the dataset
    using 5 bins. A LOF is calculated per predicate for each numerical literal, removing outliers prior to binning.

    Args:
        data (Data): The Data object representing the dataset.
        **kwargs: Additional keyword arguments.

    Returns:
        Data: The augmented Data object with the number literals binned using 5 LOF bins.

    """
    return bin_numbers(data=data, num_bins=5, use_lof=True)


def bin_numbers_lof_10(data: Data, **kwargs) -> Data:
    """
    GDA approach, binning number literals in a KG into 10 bins per predicate. Removing outliers detected with LOF.

    This method takes a Data object representing a KG and performs binning on numerical literals in the dataset
    using 10 bins. A LOF is calculated per predicate for each numerical literal, removing outliers prior to binning.

    Args:
        data (Data): The Data object representing the dataset.
        **kwargs: Additional keyword arguments.

    Returns:
        Data: The augmented Data object with the number literals binned using 10 LOF bins.

    """
    return bin_numbers(data=data, num_bins=10, use_lof=True)


def bin_numbers_lof_100(data: Data, **kwargs) -> Data:
    """
    GDA approach, binning number literals in a KG into 100 bins per predicate. Removing outliers detected with LOF.

    This method takes a Data object representing a KG and performs binning on numerical literals in the dataset
    using 100 bins. A LOF is calculated per predicate for each numerical literal, removing outliers prior to binning.

    Args:
        data (Data): The Data object representing the dataset.
        **kwargs: Additional keyword arguments.

    Returns:
        Data: The augmented Data object with the number literals binned using 100 LOF bins.

    """
    return bin_numbers(data=data, num_bins=100, use_lof=True)


def bin_numbers_percentage_1(data, **kwargs) -> Data:
    """
    GDA approach, binning number literals in a KG into n bins, where n is 1 % of unique values, per predicate.

    This method takes a Data object representing a KG and performs binning on numerical literals in the dataset
    using 1 % of unique values as the number of bins to create.

    Args:
        data (Data): The Data object representing the dataset.
        **kwargs: Additional keyword arguments.

    Returns:
        Data: The augmented Data object with the number literals binned using 1 % bins.

    """
    return bin_numbers(data, num_bins=1, num_bins_as_percent=True)


def bin_numbers_percentage_5(data, **kwargs) -> Data:
    """
    GDA approach, binning number literals in a KG into n bins, where n is 5 % of unique values, per predicate.

    This method takes a Data object representing a KG and performs binning on numerical literals in the dataset
    using 5 % of unique values as the number of bins to create.

    Args:
        data (Data): The Data object representing the dataset.
        **kwargs: Additional keyword arguments.

    Returns:
        Data: The augmented Data object with the number literals binned using 5 % bins.

    """
    return bin_numbers(data, num_bins=5, num_bins_as_percent=True)


def bin_numbers_percentage_10(data, **kwargs) -> Data:
    """
    GDA approach, binning number literals in a KG into n bins, where n is 10 % of unique values, per predicate.

    This method takes a Data object representing a KG and performs binning on numerical literals in the dataset
    using 10 % of unique values as the number of bins to create.

    Args:
        data (Data): The Data object representing the dataset.
        **kwargs: Additional keyword arguments.

    Returns:
        Data: The augmented Data object with the number literals binned using 10 % bins.

    """
    return bin_numbers(data, num_bins=10, num_bins_as_percent=True)


def one_entity(data: Data, **kwargs) -> Data:
    """
    GDA approach, replacing literal values on a per predicate basis with an uninformative node.

    This method takes a Data object representing a KG creates new triples,
    replacing literal values with an uninformative node. This approach ich conducted to investigate the importance
    of literal values and predicates in triples with literals as objects.

    Args:
        data (Data): The Data object representing the dataset.
        **kwargs: Additional keyword arguments.

    Returns:
        Data: The augmented Data object with the literals transformed into one entity per predicate.

    """
    rr = get_relevant_relations(data, ALL_LITERALS)
    for r in rr:
        df = data.triples[data.triples[:, 1] == r]
        new_df = df.clone().detach()

        new_df[:, 1] = data.num_relations
        new_df[:, 2] = data.num_entities

        data.i2r.append(
            f"https://master-thesis.com/relations#one-relation-{r}"
        )
        data.r2i[f"https://master-thesis.com/relations#one-relation-{r}"] = (
            data.num_relations
        )

        data.i2e.append(
            (
                f"https://master-thesis.com/entitys#one-literal-{r}",
                "preprocessed",
            )
        )
        data.e2i[
            (
                f"https://master-thesis.com/entitys#one-literal-{r}",
                "preprocessed",
            )
        ] = data.num_entities

        data.triples = torch.cat((data.triples, new_df), 0)
        data.num_relations += 1
        data.num_entities += 1
    return data


def bin_numbers_hierarchically(
    data: Data, list_num_bins=[5, 10, 100], **kwargs
) -> Data:
    """
    Bin numbers in the dataset hierarchically.

    This method takes a Data object representing a KG and performs hierarchical binning on numerical literals in the dataset.
    On a per predicate base, the numbers are binned using a list of specified number of bins (list_num_bins).
    Forming multiple layers of ever increasing numbers of bins

    Args:
        data (Data): The Data object representing the dataset.
        list_num_bins (List[int], optional): The list of number of bins for hierarchical binning. Defaults to [5, 10, 100].
        **kwargs: Additional keyword arguments.

    Returns:
        Data: The updated Data object with the numbers binned hierarchically.

    """
    relevent_relations = get_relevant_relations(
        data, relevant_types=RDF_NUMBER_TYPES
    )

    # Add bin specific nodes to KG
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

    if f"{URI_PREFIX}predicat#downBin" not in data.r2i:
        p = f"{URI_PREFIX}predicat#downBin"
        new_id = len(data.i2r)
        data.r2i[p] = new_id
        data.i2r.append(p)
        data.num_relations += 1

        p = f"{URI_PREFIX}predicat#upBin"
        new_id = len(data.i2r)
        data.r2i[p] = new_id
        data.i2r.append(p)
        data.num_relations += 1

    for num_bins in list_num_bins:
        for r in relevent_relations:
            p = f"{URI_PREFIX}predicat#binning{r}-num_bins{num_bins}"
            new_id = len(data.i2r)
            data.r2i[p] = new_id
            data.i2r.append(p)
            data.num_relations += 1

    # Apply binning on a per predicate basis:
    for relation in relevent_relations:
        for i in range(len(list_num_bins)):
            num_bins = list_num_bins[i]
            sub_df = encode_number_sublist(
                data.triples[data.triples[:, 1] == relation], data.i2e
            )

            for b in range(num_bins):
                o = (
                    f"{URI_PREFIX}entity#binning{b+1}-relation{relation}-num_bins{num_bins}",
                    f"{URI_PREFIX}datatype#bin",
                )
                new_id = len(data.i2e)
                data.e2i[o] = new_id
                data.i2e.append(o)
                data.num_entities += 1
                prev_iri = (
                    f"{URI_PREFIX}entity#binning{b}-relation{relation}-num_bins{num_bins}",
                    f"{URI_PREFIX}datatype#bin",
                )
                if prev_iri in data.e2i:
                    data = add_triple(
                        data, o, f"{URI_PREFIX}predicat#prevBin", prev_iri
                    )
                    data = add_triple(
                        data, prev_iri, f"{URI_PREFIX}predicat#nextBin", o
                    )
                if i > 0:
                    upper_bin_nr = (b // list_num_bins[i - 1]) + 1
                    upper_iri = (
                        f"{URI_PREFIX}entity#binning{upper_bin_nr}-relation{relation}-num_bins{list_num_bins[i-1]}",
                        f"{URI_PREFIX}datatype#bin",
                    )
                    data = add_triple(
                        data, o, f"{URI_PREFIX}predicat#upBin", upper_iri
                    )
                    data = add_triple(
                        data, upper_iri, f"{URI_PREFIX}predicat#downBin", o
                    )

            augmented_df = data.triples.clone()
            augmented_df = augmented_df[augmented_df[:, 1] == relation]

            # numpy is used here since torch.histc was not working for some reason.
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
                        f"{URI_PREFIX}entity#binning{int(t)}-relation{relation}-num_bins{num_bins}",
                        f"{URI_PREFIX}datatype#bin",
                    )
                ]
            )

            predicat_mapping = np.vectorize(
                lambda t: data.r2i[
                    f"{URI_PREFIX}predicat#binning{relation}-num_bins{num_bins}"
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


def bin_numbers_hierarchically_5_10_100(
    data: Data, list_num_bins=[5, 10, 100], **kwargs
) -> Data:
    """
    GDA approach, Bin numbers in the KG hierarchically within 3 levels using 5, 10 and 100 bins.

    This method takes a Data object representing a KG and performs hierarchical binning on numerical literals in the KG.
    The numbers are binned hierarchically in 3 levels based on the specified list of number of bins
    (list_num_bins), being 5, 10 and 100 bins.


    Args:
        data (Data): The Data object representing the dataset.
        list_num_bins (List[int], optional): The list of number of bins for hierarchical binning. Defaults to [5, 10, 100].
        **kwargs: Additional keyword arguments.

    Returns:
        Data: The updated Data object with the numbers binned hierarchically using 5_10_100 binning.
    """
    return bin_numbers_hierarchically(data, list_num_bins=list_num_bins)


def bin_numbers_hierarchically_5lvl_binary(
    data: Data, list_num_bins=[1, 2, 4, 8, 16], **kwargs
) -> Data:
    """
    GDA approach, Bin numbers in the KG hierarchically within 5 levels using 1, 2, 4, 8 and 16, bins.

    This method takes a Data object representing a KG and performs hierarchical binning on numerical literals in the KG.
    The numbers are binned hierarchically in 5 levels based on the specified list of number of bins
    (list_num_bins), being 1, 2, 4, 8 and 16 bins.


    Args:
        data (Data): The Data object representing the dataset.
        list_num_bins (List[int], optional): The list of number of bins for hierarchical binning. Defaults to [1, 2, 4, 8, 16].
        **kwargs: Additional keyword arguments.

    Returns:
        Data: The updated Data object with the numbers binned hierarchically using 5lvl_binary binning.
    """
    return bin_numbers_hierarchically(data, list_num_bins=list_num_bins)


def altering_bins(data: Data, num_bins=5, **kwargs):
    """
    Bin numbers in the dataset with alternating boundries.

    This method takes a Data object representing a KG and performs overlapping binning on numerical literals in the dataset.
    On a per predicate base, the numbers are binned using a specified number of bins (num_bins),
    where n +1 bins are calculated and their boundaries gets combined.


    Args:
        data (Data): The Data object representing the dataset.
        num_bins (int): The list of number of bins for overlapping binning. Defaults to 5.
        **kwargs: Additional keyword arguments.

    Returns:
        Data: The updated Data object with the numbers binned overlapping.

    """
    relevent_relations = get_relevant_relations(
        data, relevant_types=RDF_NUMBER_TYPES
    )

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

    for r in relevent_relations:
        p = f"{URI_PREFIX}predicat#binning{r}"
        new_id = len(data.i2r)
        data.r2i[p] = new_id
        data.i2r.append(p)
        data.num_relations += 1

    for relation in relevent_relations:
        sub_df = encode_number_sublist(
            data.triples[data.triples[:, 1] == relation], data.i2e
        )

        for b in range(num_bins):
            o = (
                f"{URI_PREFIX}entity#binning{b+1}-relation{relation}",
                f"{URI_PREFIX}datatype#bin",
            )
            new_id = len(data.i2e)
            data.e2i[o] = new_id
            data.i2e.append(o)
            data.num_entities += 1
            prev_iri = (
                f"{URI_PREFIX}entity#binning{b}-relation{relation}",
                f"{URI_PREFIX}datatype#bin",
            )
            if prev_iri in data.e2i:
                data = add_triple(
                    data, o, f"{URI_PREFIX}predicat#prevBin", prev_iri
                )
                data = add_triple(
                    data, prev_iri, f"{URI_PREFIX}predicat#nextBin", o
                )

        augmented_df = data.triples.clone()
        augmented_df = augmented_df[augmented_df[:, 1] == relation]

        # numpy is used here since torch.histc was not working for some reason.
        sub_df = torch.cat(  # put bins and sub_df together
            (
                sub_df,
                torch.from_numpy(  # get numpy solutions back
                    np.digitize(  # assign for each value in sub_df the corresponding bin
                        sub_df[:, 1],
                        np.histogram(  # calculate n bins based on values in sub_df
                            sub_df[:, 1], num_bins + 1
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

        alternating_bins = []
        rel_id = data.r2i[f"{URI_PREFIX}predicat#binning{relation}"]
        for triple in sub_df:
            if triple[2] <= num_bins:
                alternating_bins.append(
                    [
                        int(triple[0]),
                        rel_id,
                        data.e2i[
                            (
                                f"{URI_PREFIX}entity#binning{int(triple[2])}-relation{relation}",
                                f"{URI_PREFIX}datatype#bin",
                            )
                        ],
                    ]
                )
            if triple[2] > 1:
                alternating_bins.append(
                    [
                        int(triple[0]),
                        rel_id,
                        data.e2i[
                            (
                                f"{URI_PREFIX}entity#binning{int(triple[2])-1}-relation{relation}",
                                f"{URI_PREFIX}datatype#bin",
                            )
                        ],
                    ]
                )

        alternating_bins = torch.tensor(alternating_bins, dtype=torch.int32)

        object_mapping = np.vectorize(
            lambda t: data.e2i[
                (
                    f"{URI_PREFIX}entity#binning{int(t)}-relation{relation}",
                    f"{URI_PREFIX}datatype#bin",
                )
            ]
        )

        predicat_mapping = np.vectorize(
            lambda t: data.r2i[f"{URI_PREFIX}predicat#binning{relation}"]
        )

        data.triples = torch.cat((data.triples, alternating_bins), 0)

    return data


def alternating_bins_10(data: Data, num_bins=10, **kwargs) -> Data:
    """
    GDA approach, Bin numbers in the KG with 10 overlapping boundaries.

    This method takes a Data object representing a KG and performs overlapping binning on numerical literals in the KG.
    The numbers are binned with 10 overlapping boundaries,
    where n +1 bins are calculated and their boundaries gets combined.


    Args:
        data (Data): The Data object representing the dataset.
        num_bins (int): The list of number of bins for overlapping binning. Defaults to 10.
        **kwargs: Additional keyword arguments.

    Returns:
        Data: The updated Data object with the numbers binned hierarchically using 10 overlapping bins.
    """
    return altering_bins(data, num_bins=num_bins)


def alternating_bins_100(data: Data, num_bins=100, **kwargs) -> Data:
    """
    GDA approach, Bin numbers in the KG with 100 overlapping boundaries.

    This method takes a Data object representing a KG and performs overlapping binning on numerical literals in the KG.
    The numbers are binned with 100 overlapping boundaries,
    where n +1 bins are calculated and their boundaries gets combined.


    Args:
        data (Data): The Data object representing the dataset.
        num_bins (int): The list of number of bins for overlapping binning. Defaults to 100.
        **kwargs: Additional keyword arguments.

    Returns:
        Data: The updated Data object with the numbers binned hierarchically using 100 overlapping bins.
    """
    return altering_bins(data, num_bins=num_bins)
