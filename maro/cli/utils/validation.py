# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import operator
from functools import reduce

from deepdiff import DeepDiff

from maro.utils.exception.cli_exception import CliException


def validate_and_fill_dict(template_dict: dict, actual_dict: dict, optional_key_to_value: dict):
    deep_diff = DeepDiff(template_dict,
                         actual_dict).to_dict()

    missing_keys = deep_diff.get('dictionary_item_removed', [])
    for key in missing_keys:
        if key not in optional_key_to_value:
            raise CliException(
                f"Invalid deployment: key {key} not found")
        else:
            set_in_dict(
                actual_dict,
                get_map_list(deep_diff_str=key),
                optional_key_to_value[key]
            )


def set_in_dict(data_dict: dict, map_list: list, value):
    get_from_dict(data_dict, map_list[:-1])[map_list[-1]] = value


def get_map_list(deep_diff_str: str):
    deep_diff_str = deep_diff_str.replace("root", "")
    deep_diff_str = deep_diff_str.replace("['", "")
    deep_diff_str = deep_diff_str.strip("']")
    return deep_diff_str.split("']")


def get_from_dict(data_dict: dict, map_list: list):
    return reduce(operator.getitem, map_list, data_dict)
