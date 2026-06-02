# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0


import pandas as pd
from rdflib import RDFS, Graph, Literal, XSD
from kge_experiments.config import (
    EXEKGS_NT_PATH,
    EXEKGS_W_MLSEAKG_NT_PATH,
    EXEKGS_MKGA_NT_PATH,
    EXEKGS_W_MLSEAKG_MKGA_NT_PATH,
    OUTPUT_DIR,
)


# Function to count literals
def count_literals(graph):
    num_literals = 0
    num_numeric_literals = 0
    num_string_literals = 0
    num_other_literals = 0
    types = set()
    for obj in graph.objects():
        if isinstance(obj, Literal):
            num_literals += 1
            if obj.datatype in [
                XSD.integer,
                XSD.float,
                XSD.double,
                XSD.decimal,
                XSD.int,
            ]:
                num_numeric_literals += 1
            elif obj.datatype in [XSD.string, RDFS.Literal] or obj.language:
                num_string_literals += 1
            else:
                if obj.datatype not in types:
                    print(f"New datatype: {obj.datatype}")
                types.add(obj.datatype)
                num_other_literals += 1
    return num_literals, num_numeric_literals, num_string_literals, num_other_literals


def count_data_attributes(graph):
    data_attributes = set()
    for s, p, o in graph:
        if isinstance(o, Literal):
            data_attributes.add(p)
    return len(data_attributes)


if __name__ == "__main__":
    # Iterate over all .nt files in the directory
    # for nt_path in [
    #     EXEKGS_NT_PATH,
    #     EXEKGS_W_MLSEAKG_NT_PATH,
    #     # EXEKGS_MKGA_NT_PATH,
    #     # EXEKGS_W_MLSEAKG_MKGA_NT_PATH,
    # ]:
    #     # Load the .nt file
    #     g = Graph()
    #     g.parse(nt_path, format="nt")

    #     # Count triples
    #     num_triples = len(g)

    #     # Count unique entities (subjects and objects)
    #     unique_entities = set(g.subjects()).union(set(g.objects()))
    #     num_unique_entities = len(unique_entities)

    #     # Count unique relations (predicates)
    #     unique_relations = set(g.predicates())
    #     num_unique_relations = len(unique_relations)

    #     # Count literals, numeric literals, and string literals
    #     num_literals, num_numeric_literals, num_string_literals, num_other_literals = (
    #         count_literals(g)
    #     )

    #     num_data_attributes = count_data_attributes(g)

    #     print(f"File: {nt_path}")
    #     print(f"Number of triples: {num_triples}")
    #     print(f"Number of entities: {num_unique_entities}")
    #     print(f"Number of relations: {num_unique_relations}")
    #     print(f"Number of attributes: {num_data_attributes}")
    #     print(f"Number of literals: {num_literals}")
    #     print(f"Number of numeric literals: {num_numeric_literals}")
    #     print(f"Number of string literals: {num_string_literals}")
    #     print(f"Number of other literals: {num_other_literals}")
    #     print()

    #     pd.DataFrame(
    #         {
    #             "File": [nt_path.name],
    #             "Triples": [num_triples],
    #             "Entities": [num_unique_entities],
    #             "Relations": [num_unique_relations],
    #             "Literals": [num_literals],
    #             "Numeric Literals": [num_numeric_literals],
    #             "String Literals": [num_string_literals],
    #             "Other Literals": [num_other_literals],
    #         }
    #     ).to_csv(OUTPUT_DIR / "dataset_stats.csv", mode="a", index=False)

    for nt_path in [
        EXEKGS_MKGA_NT_PATH,
        EXEKGS_W_MLSEAKG_MKGA_NT_PATH,
    ]:
        # Load the .nt file
        g = Graph()
        g.parse(nt_path, format="nt")

        unique_relations = set()
        unique_entities = set()
        num_triples = 0
        for s, p, o in g:
            p_is_mkga = str(p).startswith(
                "http://multimodal-knowledge-graph-augmentation.com"
            )
            s_is_mkga = str(s).startswith(
                "http://multimodal-knowledge-graph-augmentation.com"
            )
            o_is_mkga = str(o).startswith(
                "http://multimodal-knowledge-graph-augmentation.com"
            )
            if p_is_mkga:
                unique_relations.add(p)

            if s_is_mkga:
                unique_entities.add(s)

            if o_is_mkga:
                unique_entities.add(o)

            if s_is_mkga or o_is_mkga or p_is_mkga:
                num_triples += 1

        num_unique_relations = len(unique_relations)
        num_unique_entities = len(unique_entities)

        # # Count literals, numeric literals, and string literals
        # num_literals, num_numeric_literals, num_string_literals, num_other_literals = (
        #     count_literals(g)
        # )

        # num_data_attributes = count_data_attributes(g)

        print(f"File: {nt_path}")
        print(f"Number of MKGA triples: {num_triples}")
        print(f"Number of MKGA entities: {num_unique_entities}")
        print(f"Number of MKGA relations: {num_unique_relations}")
        # print(f"Number of attributes: {num_data_attributes}")
        # print(f"Number of literals: {num_literals}")
        # print(f"Number of numeric literals: {num_numeric_literals}")
        # print(f"Number of string literals: {num_string_literals}")
        # print(f"Number of other literals: {num_other_literals}")
        print()

        pd.DataFrame(
            {
                "File": [nt_path.name],
                "Triples": [num_triples],
                "Entities": [num_unique_entities],
                "Relations": [num_unique_relations],
                "Literals": [None],
                "Numeric Literals": [None],
                "String Literals": [None],
                "Other Literals": [None],
            }
        ).to_csv(OUTPUT_DIR / "dataset_stats.csv", mode="a", index=False)
