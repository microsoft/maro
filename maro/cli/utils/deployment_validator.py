# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import operator
from functools import reduce

from deepdiff import DeepDiff

from maro.utils.exception.cli_exception import InvalidDeploymentTemplateError


class DeploymentValidator:
    @staticmethod
    def validate_and_fill_dict(template_dict: dict, actual_dict: dict, optional_key_to_value: dict) -> None:
        """Validate incoming actual_dict with template_dict, and fill optional keys to the template.

        We use deepDiff to find missing keys in the actual_dict, see
        https://deepdiff.readthedocs.io/en/latest/diff.html#deepdiff-reference for reference.

        Args:
            template_dict (dict): template dict, we only need the layer structure of keys here, and ignore values.
            actual_dict (dict): the actual dict with values, may miss some keys.
            optional_key_to_value (dict): mapping of optional keys to values.

        Returns:
            None.
        """
        deep_diff = DeepDiff(template_dict, actual_dict).to_dict()

        missing_key_strs = deep_diff.get('dictionary_item_removed', [])
        for missing_key_str in missing_key_strs:
            if missing_key_str not in optional_key_to_value:
                raise InvalidDeploymentTemplateError(f"Key '{missing_key_str}' not found.")
            else:
                DeploymentValidator._set_value(
                    original_dict=actual_dict,
                    key_list=DeploymentValidator._get_parent_to_child_key_list(deep_diff_str=missing_key_str),
                    value=optional_key_to_value[missing_key_str]
                )

    @staticmethod
    def _set_value(original_dict: dict, key_list: list, value) -> None:
        """Set the value to the original dict based on the key_list.

        Args:
            original_dict (dict): original dict that needs to be modified.
            key_list (list): the parent to child path of keys, which describes that position of the value.
            value: the value needs to be set.

        Returns:
            None.
        """
        DeploymentValidator._get_sub_structure_of_dict(original_dict, key_list[:-1])[key_list[-1]] = value

    @staticmethod
    def _get_parent_to_child_key_list(deep_diff_str: str) -> list:
        """Get parent to child key list by parsing the deep_diff_str.

        Args:
            deep_diff_str (str): a specially defined string that indicate the position of the key.
                e.g. "root['a']['b']" -> {"a": {"b": value}}.

        Returns:
            list: the parent to child path of keys.
        """

        deep_diff_str = deep_diff_str.strip("root['")
        deep_diff_str = deep_diff_str.strip("']")
        return deep_diff_str.split("']['")

    @staticmethod
    def _get_sub_structure_of_dict(original_dict: dict, key_list: list) -> dict:
        """Get sub structure of dict from original_dict and key_list using reduce.

        Args:
            original_dict (dict): original dict that needs to be modified.
            key_list (list): the parent to child path of keys, which describes that position of the value.

        Returns:
            dict: sub structure of the original_dict.
        """

        return reduce(operator.getitem, key_list, original_dict)
