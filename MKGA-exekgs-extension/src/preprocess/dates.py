import pandas as pd
import numpy as np
import torch

from utils import (
    get_relevant_relations,
    RDF_DATE_TYPES,
    add_triple,
    URI_PREFIX,
    Data,
)


def append_date_features(data: Data, **kwargs) -> Data:
    """
    GDA approach, appending date features to the dataset.

    This method takes a Data object representing a KG and appends date features to the set of triples.

    Args:
        data (Data): The Data object representing the KG.
        **kwargs: Additional keyword arguments.

    Returns:
        Data: The updated Data object with the appended date features.

    """

    date_features = [
        "day_of_month",
        "day_of_week",
        "month_of_year",
        "quarter_of_year",
        "year",
    ]
    feature_ranges = {
        "day_of_week": [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ],
        "month_of_year": [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
        ],
        "quarter_of_year": ["1", "2", "3", "4"],
    }

    relevent_relations = get_relevant_relations(data, RDF_DATE_TYPES)

    if f"{URI_PREFIX}predicat#prevDate" not in data.r2i:
        p = f"{URI_PREFIX}predicat#prevDate"
        new_id = len(data.i2r)
        data.r2i[p] = new_id
        data.i2r.append(p)
        data.num_relations += 1

        p = f"{URI_PREFIX}predicat#nextDate"
        new_id = len(data.i2r)
        data.r2i[p] = new_id
        data.i2r.append(p)
        data.num_relations += 1

    for relation in relevent_relations:
        for feature in date_features:
            p = f"{URI_PREFIX}predicat#aug_date_{relation}_{feature}"
            new_id = len(data.i2r)
            data.r2i[p] = new_id
            data.i2r.append(p)
            data.num_relations += 1

    for relation in relevent_relations:
        df = pd.DataFrame(data.triples[data.triples[:, 1] == relation], columns=["s", "p", "o"])  # type: ignore
        df["t"] = df["o"].apply(lambda x: data.i2e[x][0])
        df["t"] = pd.to_datetime(df["t"], errors="coerce")
        df = df[df["t"].notnull()]
        df["day_of_month"] = df["t"].apply(lambda x: str(x.strftime("%d")))
        df["day_of_week"] = df["t"].apply(lambda x: str(x.strftime("%A")))
        df["month_of_year"] = df["t"].apply(lambda x: str(x.strftime("%m")))
        df["quarter_of_year"] = df["t"].apply(
            lambda x: str(((int(x.strftime("%m")) - 1) // 4) + 1)
        )
        df["year"] = df["t"].apply(lambda x: str(x.strftime("%Y")))

        for feature in date_features:
            if feature in feature_ranges:
                for i in range(len(feature_ranges[feature])):
                    entry = feature_ranges[feature][i]
                    o = (
                        f"{URI_PREFIX}entity#{feature}{entry}-relation{relation}",
                        f"{URI_PREFIX}datatype#feature",
                    )
                    new_id = len(data.i2e)
                    data.e2i[o] = new_id
                    data.i2e.append(o)
                    data.num_entities += 1
                    if i > 0:
                        prev_o = (
                            f"{URI_PREFIX}entity#{feature}{feature_ranges[feature][i-1]}-relation{relation}",
                            f"{URI_PREFIX}datatype#feature",
                        )
                        data = add_triple(
                            data, o, f"{URI_PREFIX}predicat#prevDate", prev_o
                        )
                        data = add_triple(
                            data, prev_o, f"{URI_PREFIX}predicat#nextDate", o
                        )
            else:
                for f in df[feature].unique():
                    o = (
                        f"{URI_PREFIX}entity#{feature}{f}-relation{relation}",
                        f"{URI_PREFIX}datatype#feature",
                    )
                    new_id = len(data.i2e)
                    data.e2i[o] = new_id
                    data.i2e.append(o)
                    data.num_entities += 1

        for feature in date_features:
            df["new_o"] = df[feature].apply(
                lambda f: data.e2i[
                    (
                        f"{URI_PREFIX}entity#{feature}{f}-relation{relation}",
                        f"{URI_PREFIX}datatype#feature",
                    )
                ]
                if (not pd.isnull(f)) & (f != "")
                else np.nan
            )
            df["new_p"] = df[feature].apply(
                lambda f: data.r2i[
                    f"{URI_PREFIX}predicat#aug_date_{relation}_{feature}"
                ]
                if (not pd.isnull(f)) & (f != "")
                else np.nan
            )
            ten = torch.tensor(
                df[(df["new_o"].notnull())][
                    ["s", "new_p", "new_o"]
                ].values.astype(np.int32),
                dtype=torch.int32,
            )
            data.triples = torch.cat((data.triples, ten), 0)

    return data


def bin_dates(data: Data, num_bins=100, **kwargs) -> Data:
    """
    GDA approach, binning date literals into bins on a per predicate level.

    This method takes a Data object representing a KG and bins the unix timestamp into 100 bins for each predicate

    Args:
        data (Data): The Data object representing the KG.
        num_bins (int, optional): The number of bins to use for binning. Defaults to 100.
        **kwargs: Additional keyword arguments.

    Returns:
        Data: The updated Data object with the appended date bins.

    """
    relevent_relations = get_relevant_relations(data, RDF_DATE_TYPES)

    if f"{URI_PREFIX}predicat#prevDate" not in data.r2i:
        p = f"{URI_PREFIX}predicat#prevDate"
        new_id = len(data.i2r)
        data.r2i[p] = new_id
        data.i2r.append(p)
        data.num_relations += 1

        p = f"{URI_PREFIX}predicat#nextDate"
        new_id = len(data.i2r)
        data.r2i[p] = new_id
        data.i2r.append(p)
        data.num_relations += 1

    for relation in relevent_relations:
        p = f"{URI_PREFIX}predicat#bin_date_{relation}"
        new_id = len(data.i2r)
        data.r2i[p] = new_id
        data.i2r.append(p)
        data.num_relations += 1

    for relation in relevent_relations:
        for i in range(num_bins):
            # entry = feature_ranges[feature][i]
            o = (
                f"{URI_PREFIX}entity#{i+1}-relation{relation}",
                f"{URI_PREFIX}datatype#bin",
            )
            new_id = len(data.i2e)
            data.e2i[o] = new_id
            data.i2e.append(o)
            data.num_entities += 1
            if i > 0:
                prev_o = (
                    f"{URI_PREFIX}entity#{i}-relation{relation}",
                    f"{URI_PREFIX}datatype#bin",
                )
                data = add_triple(
                    data, o, f"{URI_PREFIX}predicat#prevDate", prev_o
                )
                data = add_triple(
                    data, prev_o, f"{URI_PREFIX}predicat#nextDate", o
                )

        df = pd.DataFrame(
            data.triples[data.triples[:, 1] == relation],  # type: ignore
            columns=["s", "p", "o"],
        )
        df["t"] = df["o"].apply(lambda x: data.i2e[x][0])
        df["t"] = pd.to_datetime(df["t"], errors="coerce")
        df = df[df["t"].notnull()]
        df["t"] = df["t"].values.astype("int")

        df["bins"] = torch.from_numpy(  # get numpy solutions back
            np.digitize(  # assign for each value in sub_df the corresponding bin
                df["t"].values,
                np.histogram(  # calculate n bins based on values in sub_df
                    df["t"].values, num_bins
                )[1][:-1],
            )
        ).reshape(
            -1, 1
        )  # transfrom x tensor into (x,1) tensor to fit (x,2) shape of sub_df

        df["new_o"] = df["bins"].apply(
            lambda x: data.e2i[
                (
                    f"{URI_PREFIX}entity#{x}-relation{relation}",
                    f"{URI_PREFIX}datatype#bin",
                )
            ]
        )
        df["new_p"] = df["bins"].apply(
            lambda x: data.r2i[f"{URI_PREFIX}predicat#bin_date_{relation}"]
        )
        ten = torch.tensor(
            df[(df["new_o"].notnull())][["s", "new_p", "new_o"]].values.astype(
                np.int32
            ),
            dtype=torch.int32,
        )
        data.triples = torch.cat((data.triples, ten), 0)
    return data
