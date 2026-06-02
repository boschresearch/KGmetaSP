# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0


from typing import Tuple
import numpy as np
from scipy.spatial.distance import cosine, euclidean, cityblock


def calculate_emb_similarities(avg_embeddings: dict) -> Tuple[list, list, list, list]:
    """Calculate cosine similarities between embeddings and sort them."""
    cosine_sims, euclidean_dist_sims, manhattan_dist_sims, dot_product_sims = (
        [],
        [],
        [],
        [],
    )

    def add_similarities(
        dataset_id1,
        dataset_id2,
        cosine_sim,
        euclidean_dist_sim,
        manhattan_dist_sim,
        dot_product_sim,
    ):
        for sims, sim in zip(
            [cosine_sims, euclidean_dist_sims, manhattan_dist_sims, dot_product_sims],
            [cosine_sim, euclidean_dist_sim, manhattan_dist_sim, dot_product_sim],
        ):
            sims.append(
                (
                    dataset_id1,
                    dataset_id2,
                    sim,
                )
            )

    dataset_ids = sorted(list(avg_embeddings.keys()))

    for i, id1 in enumerate(dataset_ids):
        for j in range(i + 1, len(dataset_ids)):
            id2 = dataset_ids[j]
            if avg_embeddings[id1] is None or avg_embeddings[id2] is None:
                add_similarities(id1, id2, None, None, None, None)
                continue

            cosine_sim = 1 - cosine(avg_embeddings[id1], avg_embeddings[id2])
            euclidean_dist_sim = 1 / (
                1 + euclidean(avg_embeddings[id1], avg_embeddings[id2])
            )
            manhattan_dist_sim = 1 / (
                1 + cityblock(avg_embeddings[id1], avg_embeddings[id2])
            )
            dot_product_sim = np.dot(avg_embeddings[id1], avg_embeddings[id2])

            add_similarities(
                id1,
                id2,
                cosine_sim,
                euclidean_dist_sim,
                manhattan_dist_sim,
                dot_product_sim,
            )

    print(f"Calculated similarities for {len(cosine_sims)} dataset pairs.")
    return cosine_sims, euclidean_dist_sims, manhattan_dist_sims, dot_product_sims


def get_nums_of_common_items(items_dict: dict) -> list:
    """Calculate cosine similarities between embeddings and sort them."""
    nums_of_common_items = []
    dataset_ids = sorted(list(items_dict.keys()))

    for i, id1 in enumerate(dataset_ids):
        for j in range(i + 1, len(dataset_ids)):
            id2 = dataset_ids[j]
            num_of_common_items = len(set(items_dict[id1]) & set(items_dict[id2]))
            nums_of_common_items.append((id1, id2, num_of_common_items))

    return nums_of_common_items
