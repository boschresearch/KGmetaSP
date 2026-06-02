# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import pandas as pd


import gzip
import pathlib
import re
import tarfile


def create_kgbench_dataset_from_nt(
    nt_path: pathlib.Path, output_tgz_path: pathlib.Path
):
    print(f"Creating KGBench dataset from NT file {nt_path} ...")

    output_dir = output_tgz_path.parent
    nodes_path = output_dir / "nodes.int.csv"
    relations_path = output_dir / "relations.int.csv"
    nodetypes_path = output_dir / "nodetypes.int.csv"
    triples_path = output_dir / "triples.int.csv.gz"
    training_path = output_dir / "training.int.csv"
    validation_path = output_dir / "validation.int.csv"
    testing_path = output_dir / "testing.int.csv"

    entities = set()
    relations = set()
    triples = []
    literals = set()

    with open(nt_path, "r", encoding="utf-8") as nt_file:
        for line in nt_file:
            # Replace the second occurrence of a space with a tab
            line = re.sub(r" ", "\t", line, count=2)
            # Remove any spaces followed by a period
            line = re.sub(r" \.", "", line)
            line = line[:-1]  # Remove the newline character

            if line == "":  # Skip empty lines
                continue

            triple_parts = line.split("\t")
            triple_parts = [
                part[1:-1] if part.startswith("<") else part for part in triple_parts
            ]

            entities.add(triple_parts[0])
            if triple_parts[2].startswith('"'):
                # Extract literal value and type
                if '"^^<' not in triple_parts[2]:
                    print(
                        f"Literal without type: {triple_parts[2]}. Assuming string ..."
                    )
                    literals.add(
                        (
                            triple_parts[2][1:-1],
                            "http://www.w3.org/2001/XMLSchema#string",
                        )
                    )
                else:
                    literal_value = triple_parts[2].split('"^^<')[0][1:]
                    literal_type = triple_parts[2].split('"^^<')[1][:-1]
                    literals.add((literal_value, literal_type))
            else:
                entities.add(triple_parts[2])
            relations.add(triple_parts[1])

            triples.append(triple_parts)

    # Create indices for entities, relations, and literals
    entity_to_index = {entity: idx for idx, entity in enumerate(entities)}
    relation_to_index = {relation: idx for idx, relation in enumerate(relations)}
    literal_to_index = {literal: idx for idx, literal in enumerate(literals)}

    # Combine entities and literals for nodes
    nodes = [(entity, "iri") for entity in entities] + [
        (literal[0], literal[1]) for literal in literals
    ]
    node_and_type_to_index = {(node[0], node[1]): idx for idx, node in enumerate(nodes)}

    # Save nodes and relations to CSV files
    nodes_df = pd.DataFrame(
        [
            (idx, node_and_type[1], node_and_type[0])
            for node_and_type, idx in node_and_type_to_index.items()
        ],
        columns=["index", "annotation", "label"],
    )
    nodes_df.to_csv(nodes_path, index=False)

    relations_df = pd.DataFrame(
        list(relation_to_index.items()), columns=["label", "index"]
    )
    relations_df.to_csv(relations_path, index=False)

    # Save node types to nodetypes.int.csv
    nodetypes = {node[1] for node in nodes}
    nodetypes_to_index = {nodetype: idx for idx, nodetype in enumerate(nodetypes)}
    nodetypes_df = pd.DataFrame(
        list(nodetypes_to_index.items()), columns=["annotation", "index"]
    )
    nodetypes_df.to_csv(nodetypes_path, index=False)

    # Convert triples to use indices
    indexed_triples = []
    for triple in triples:
        s_idx = node_and_type_to_index[(triple[0], "iri")]
        p_idx = relation_to_index[triple[1]]
        if triple[2].startswith('"'):
            if '"^^<' not in triple[2]:
                literal_value = triple[2][1:-1]
                literal_type = "http://www.w3.org/2001/XMLSchema#string"
            else:
                literal_value = triple[2].split('"^^<')[0][1:]
                literal_type = triple[2].split('"^^<')[1][:-1]
            o_idx = node_and_type_to_index[(literal_value, literal_type)]
        else:
            o_idx = node_and_type_to_index[(triple[2], "iri")]
        indexed_triples.append([s_idx, p_idx, o_idx])

    # Save indexed triples to triples.int.csv.gz with no headers
    triples_df = pd.DataFrame(indexed_triples)
    # triples_df.to_csv("triples.int.csv", index=False, header=False)

    with gzip.open(triples_path, "wt", encoding="utf-8") as f:
        triples_df.to_csv(f, index=False, header=False)

    # insert dummy data for training, validation, and testing as we won't work with classification
    pd.DataFrame([(0, 0), (1, 1)], columns=["index", "class"]).to_csv(
        training_path, index=False
    )
    pd.DataFrame([(2, 0), (3, 1)], columns=["index", "class"]).to_csv(
        validation_path, index=False
    )
    pd.DataFrame([(4, 0), (5, 1)], columns=["index", "class"]).to_csv(
        testing_path, index=False
    )

    # Bundle all created files into a .tgz file
    with tarfile.open(output_tgz_path, "w:gz") as tar:
        tar.add(nodes_path, arcname="nodes.int.csv")
        tar.add(relations_path, arcname="relations.int.csv")
        tar.add(nodetypes_path, arcname="nodetypes.int.csv")
        tar.add(triples_path, arcname="triples.int.csv.gz")
        tar.add(training_path, arcname="training.int.csv")
        tar.add(validation_path, arcname="validation.int.csv")
        tar.add(testing_path, arcname="testing.int.csv")

    print(f"KGBench dataset created into {output_tgz_path}")


def create_nt_from_kgbench_data(data, output_nt_path):
    """
    Writes triples (with labels, no ids) as N-triples to an .nt file.

    :param data: Data object containing the triples and entity/relation mappings.
    :param output_file: Path to the output .nt file.
    """
    with open(str(output_nt_path), "w") as f:
        for s, p, o in data.triples:
            subject = data.i2e[s][0]
            predicate = data.i2r[p]
            obj = data.i2e[o][0]

            obj_type = data.i2e[o][1]
            if obj_type in [
                "iri",
                "http://multimodal-knowledge-graph-augmentation.com/datatype#topics",
                "http://multimodal-knowledge-graph-augmentation.com/outlier",
                "http://multimodal-knowledge-graph-augmentation.com/datatype#bin",
            ]:
                obj = f"<{obj}>"
            else:
                obj = f'"{obj}"^^<{obj_type}>'

            # Format as N-triples
            nt_line = f"<{subject}> <{predicate}> {obj} .\n"
            f.write(nt_line)

    print(f"Data written to {output_nt_path}")