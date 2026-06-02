# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0


import glob
import re
from typing import Dict, List, Tuple
import networkx as nx
from rdflib import RDF, RDFS, Graph, Namespace
import pathlib
from SPARQLWrapper import N3, SPARQLWrapper

from kge_experiments.config import (
    EXEKGS_RAW_DIR,
    EXEKGS_W_MLSEAKG_NT_PATH,
    EXEKGS_NT_PATH,
    MLSEAKG_FILTERED_NT_PATH,
    MLSEAKG_GRAPHDB_ENDPOINT,
)
from kge_experiments.constants.queries import MLSEAKG_FEATURE_IRIS, MLSEAKG_FILTER_QUERY
from kge_experiments.utils.string_utils import get_openml_ids_from_file_path
import shutil

DS = Namespace(
    "https://raw.githubusercontent.com/nsai-uio/ExeKGOntology/main/ds_exeKGOntology.ttl#"
)
ML = Namespace(
    "https://raw.githubusercontent.com/nsai-uio/ExeKGOntology/main/ml_exeKGOntology.ttl#"
)

MLS = Namespace("http://www.w3.org/ns/mls#")


def load_graph_from_ttl(file_path: str) -> nx.Graph:
    """Load a graph from a TTL file and convert it to a NetworkX graph."""
    g = Graph()
    g.parse(file_path, format="ttl")
    nx_graph = nx.Graph()

    # Identify entities of type ds:DataEntity and ml:DataConcatenation
    excluded_entities = set()
    for rdf_type in [
        DS.DataEntity,
        ML.DataConcatenation,
        ML.DataInConcatenation,
        DS.Pipeline,
    ]:
        for s, p, o in g.triples((None, RDF.type, rdf_type)):
            excluded_entities.add(s)

    # Add edges to the NetworkX graph, excluding triples with the identified entities
    for s, p, o in g:
        if s not in excluded_entities and o not in excluded_entities:
            s_str = str(s)
            o_str = str(o)
            s_str = s_str.split("_pipeline")[0]
            o_str = o_str.split("_pipeline")[0]
            nx_graph.add_edge(s_str, o_str, predicate=str(p))
    return nx_graph


def load_graphs_for_task(
    task_id: int, exclude_flows_per_task: dict
) -> Dict[str, nx.Graph]:
    path_pattern = str(EXEKGS_RAW_DIR / f"task_{task_id}" / "flow_*" / "run_*.ttl")
    run_to_graph = {}
    for file_path in glob.glob(path_pattern):
        _, flow_id, run_id = get_openml_ids_from_file_path(file_path)
        if (
            task_id in exclude_flows_per_task
            and flow_id in exclude_flows_per_task[task_id]
        ):
            continue

        run_to_graph[run_id] = load_graph_from_ttl(file_path)

    if len(run_to_graph) == 0:
        print(f"No (valid) runs found for task {task_id}")
        return None

    return run_to_graph


def get_pipeline_info_for_task(
    task_id: int, exekgs_path: pathlib.Path, exclude_flows_per_task: dict
) -> Tuple[List[str], List[int], List[int]]:
    """Create a pipeline name of the format pipeline_task_{task_id}_flow_{flow_id}_run_{run_id}."""
    path_pattern = str(exekgs_path / f"task_{task_id}" / "flow_*" / "run_*.ttl")
    file_paths = glob.glob(path_pattern)

    pipeline_names = []
    flow_ids = []
    run_ids = []
    for file_path in file_paths:
        _, flow_id, run_id = get_openml_ids_from_file_path(file_path)

        if (
            task_id in exclude_flows_per_task
            and flow_id in exclude_flows_per_task[task_id]
        ):
            continue

        pipeline_names.append(f"pipeline_task_{task_id}_flow_{flow_id}_run_{run_id}")
        flow_ids.append(flow_id)
        run_ids.append(run_id)

    return pipeline_names, flow_ids, run_ids


def filter_mlseakg(dataset_ids: List[int]):
    print(f"Filtering MLSeaKG for {len(dataset_ids)} datasets ...")

    values_clause = " ".join(f"{dataset_id}" for dataset_id in dataset_ids)

    query = MLSEAKG_FILTER_QUERY.format(values_clause=values_clause)

    sparql = SPARQLWrapper(MLSEAKG_GRAPHDB_ENDPOINT)
    sparql.setQuery(query)
    sparql.setReturnFormat(N3)
    results = sparql.query().convert()

    result_graph = Graph()
    result_graph.parse(data=results, format="nt")

    result_graph.serialize(
        destination=MLSEAKG_FILTERED_NT_PATH, format="nt", encoding="utf-8"
    )


def combine_exekgs_nt_with_mlseakg_nt(dataset_ids: List[int]):
    print("Combining ExeKGS and MLSeaKG ...")

    mlseakg_g = Graph()
    mlseakg_g.parse(MLSEAKG_FILTERED_NT_PATH, format="nt")

    # Filter subjects based on a regex pattern
    regex_pattern = re.compile(r"^http://w3id.org/mlsea/openml/feature/\d+-\d+$")
    sameas_triples = []
    print("Creating sameAs triples ...")
    for i, feature_iri in enumerate(mlseakg_g.subjects()):
        if not regex_pattern.match(str(feature_iri)):
            continue

        dataset_id = feature_iri.split("/")[-1].split("-")[0]
        feature_label = mlseakg_g.value(feature_iri, RDFS.label)
        if " " in feature_label:
            print(f"Feature label {feature_label} contains spaces. Removing them ...")
            old_label = feature_label
            feature_label = feature_label.replace(" ", "")
            print(f'Old label: "{old_label}", new label: "{feature_label}"')

        sameas_triples.append(
            f"<https://raw.githubusercontent.com/nsai-uio/ExeKGOntology/main/ds_exeKGOntology.ttl#{feature_label}_dataset_{dataset_id}> <http://www.w3.org/2002/07/owl#sameAs> <{feature_iri}> ."
        )

    del mlseakg_g

    shutil.copy(EXEKGS_NT_PATH, EXEKGS_W_MLSEAKG_NT_PATH)

    # Write the combined content to a new file
    with open(EXEKGS_W_MLSEAKG_NT_PATH, "a", encoding="utf-8") as combined_file:
        with open(MLSEAKG_FILTERED_NT_PATH, "r", encoding="utf-8") as mlseakg_file:
            mlseakg_content = mlseakg_file.read()
            combined_file.write(mlseakg_content)
            del mlseakg_content

        combined_file.write("\n".join(sameas_triples))

    print("Combination complete.")
