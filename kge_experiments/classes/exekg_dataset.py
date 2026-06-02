# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0



import re
from rdflib.util import guess_format
from rdflib import Graph as rdfGraph
from igraph import Graph
from pathlib import Path

from kge_experiments.config import (
    EXEKGS_MKGA_NT_PATH,
    EXEKGS_W_MLSEAKG_MKGA_NT_PATH,
    EXEKGS_W_MLSEAKG_NT_PATH,
    EXEKGS_NT_PATH,
)


class ExeKGDataset:
    def __init__(self, use_mlseakg=False, use_mkga=False):
        if use_mlseakg:
            if use_mkga:
                nt_path = str(EXEKGS_W_MLSEAKG_MKGA_NT_PATH)
            else:
                nt_path = str(EXEKGS_W_MLSEAKG_NT_PATH)
        else:
            if use_mkga:
                nt_path = str(EXEKGS_MKGA_NT_PATH)
            else:
                nt_path = str(EXEKGS_NT_PATH)

        self.use_mlseakg = use_mlseakg

        entities, relations, exekg_de_id_to_mlseakg_feature_id = (
            self.get_kg_info_from_nt(nt_path)
        )

        self.entities = [entity[1:-1] for entity in entities]
        self.relations = [relation[1:-1] for relation in relations]
        self.exekg_de_id_to_mlseakg_feature_id = (
            exekg_de_id_to_mlseakg_feature_id if use_mlseakg else None
        )

    def get_kg_info_from_nt(self, nt_path):
        exekg_de_id_to_mlseakg_feature_id = {}
        entities = set()
        relations = set()
        regex_pattern = re.compile(r"^<http://w3id.org/mlsea/openml/feature/\d+-\d+>$")

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
                entities.add(triple_parts[0])
                entities.add(triple_parts[2])
                relations.add(triple_parts[1])

                if (
                    self.use_mlseakg
                    and regex_pattern.match(triple_parts[0])
                    and triple_parts[1]
                    == "<http://www.w3.org/2000/01/rdf-schema#label>"
                ):
                    feature_iri = triple_parts[0]
                    feature_label = triple_parts[2].split('"')[1]
                    feature_label = feature_label.replace(" ", "")
                    dataset_id = feature_iri.split("/")[-1].split("-")[0]
                    exekg_de_id_to_mlseakg_feature_id[
                        f"{feature_label}_dataset_{dataset_id}"
                    ] = feature_iri.split("/")[-1][:-1]

        return list(entities), list(relations), exekg_de_id_to_mlseakg_feature_id
