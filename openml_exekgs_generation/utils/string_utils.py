# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import ast
from typing import Any
from openml_exekgs_generation.config import (
    OPENML_PARAM_VALUES_TO_SKLEARN,
    OPENML_PARAMS_TO_NEW_SKLEARN,
    OPENML_TO_SKLEARN_CLASSES,
)


def snake_to_camel(text: str) -> str:
    """
    Converts camel-case string to snake-case
    Args:
        text: string to convert

    Returns:
        str: converted string
    """
    return "".join(x.capitalize() for x in text.lower().split("_"))


def code_param_to_exekg_param(param_name: str) -> str:
    """
    Converts a parameter name from code format to ExeKG format.

    Args:
        param_name (str): The parameter name in code format.

    Returns:
        str: The parameter name in ExeKG format.
    """
    return f"hasParam{snake_to_camel(param_name)}"


def code_method_to_exekg_method(method_name: str) -> str:
    """
    Converts a method name from code format to ExeKG format.

    Args:
        method_name (str): The method name in code format.

    Returns:
        str: The method name in ExeKG format.
    """
    if method_name.lower() == method_name:
        method_name = snake_to_camel(method_name)
    return method_name + "Method"


def openml_param_value_to_exekg_param_value(param_value: str) -> Any:
    """
    Converts a parameter value from OpenML format to ExeKG format.

    Args:
        param_value (str): The parameter value in OpenML format.

    Returns:
        Any: The parameter value in ExeKG format.
    """
    if (
        not isinstance(param_value, dict)
        and param_value in OPENML_PARAM_VALUES_TO_SKLEARN
    ):
        return OPENML_PARAM_VALUES_TO_SKLEARN[param_value.lower()]
    else:
        try:
            param_value = str(param_value)
            param_value = ast.literal_eval(param_value)
            if isinstance(param_value, float) and param_value == int(param_value):
                # convert float to int if it's actually an int, to avoid issues with ExeKG validation
                param_value = int(param_value)
            elif isinstance(
                param_value, dict
            ) and "value" in param_value:  # e.g. {'oml-python:serialized_object': 'type', 'value': 'np.float64'}
                return openml_param_value_to_exekg_param_value(param_value["value"])
            return param_value
        except (ValueError, KeyError, SyntaxError):
            return str(param_value)


def get_param_dict_from_openml_run(run, filter_by_component_id=None):
    param_settings = run.parameter_settings
    if filter_by_component_id is not None:
        param_settings = filter(
            lambda s: int(s["oml:component"]) == filter_by_component_id, param_settings
        )

    converted_param_settings = {}
    for s in param_settings:
        param_name = s["oml:name"]
        if param_name in OPENML_PARAMS_TO_NEW_SKLEARN:
            param_name = OPENML_PARAMS_TO_NEW_SKLEARN[param_name]
            if param_name is None:
                continue
        param_name = code_param_to_exekg_param(param_name)

        param_value = openml_param_value_to_exekg_param_value(s["oml:value"])
        if param_value is None:
            continue

        converted_param_settings[param_name] = param_value

    return converted_param_settings


def get_method_name_and_params_dict(run, component_obj):
    component_class = component_obj.class_name.split(".")[-1]
    if component_class in OPENML_TO_SKLEARN_CLASSES:
        component_class = OPENML_TO_SKLEARN_CLASSES[component_class]
        # print(f"Component class updated to {component_class}")

    method_name = code_method_to_exekg_method(component_class)
    method_params_dict = (
        get_param_dict_from_openml_run(run, filter_by_component_id=component_obj.id)
        if run
        else {}
    )

    return method_name, method_params_dict
