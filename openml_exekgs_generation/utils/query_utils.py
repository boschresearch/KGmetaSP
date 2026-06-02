# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from rdflib import URIRef
from exe_kg_lib.utils.query_utils import (
    get_first_query_result_if_exists,
    query_hierarchy_chain,
)
from exe_kg_lib.config import KG_SCHEMAS


def query_task_class_by_method_class(
    kg,
    method_class_iri: str,
    namespace_prefix,
):
    return kg.query(
        f"SELECT ?t WHERE {{ ?p rdfs:domain ?t ."
        f"                   ?p rdfs:range ?m ."
        f"                   ?p rdfs:subPropertyOf* {namespace_prefix}:hasMethod . }}",
        initBindings={"m": URIRef(method_class_iri)},
    )


def get_lower_and_upper_task_of_method(exe_kg, method_name: str):
    query_result = get_first_query_result_if_exists(
        query_task_class_by_method_class,
        exe_kg.input_kg,
        KG_SCHEMAS["Machine Learning"]["namespace"] + method_name,
        KG_SCHEMAS["Data Science"]["namespace_prefix"],
    )

    if query_result is None:
        print(f"Task class not found for method {method_name}")
        return None, None

    method_lower_task_iri = query_result[0]

    task_chain_query_res = list(
        query_hierarchy_chain(exe_kg.input_kg, method_lower_task_iri)
    )

    method_upper_task_name = None
    task_chain_names = [elem[0].split("#")[-1] for elem in task_chain_query_res]
    for task_name in reversed(task_chain_names):
        if task_name == "Train" or task_name == "PrepareTransformer":
            method_upper_task_name = task_name
            break

    if method_upper_task_name is None:
        print(f"No upper task found for method {method_name}")
        exit(1)

    method_lower_task_name = method_lower_task_iri.split("#")[-1]

    return method_lower_task_name, method_upper_task_name
