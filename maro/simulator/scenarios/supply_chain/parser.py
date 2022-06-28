# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from dataclasses import dataclass
from importlib import import_module
from typing import Dict, Optional

from yaml import safe_load


@dataclass
class ModuleDef:
    alias: str
    module_path: str
    class_name: str
    class_type: type


@dataclass
class DataModelDef(ModuleDef):
    name_in_frame: str


@dataclass
class EntityDef(ModuleDef):
    data_model_alias: Optional[str]


def find_class_type(module_path: str, class_name: str) -> type:
    """Find class type by module path and class name.

    Args:
        module_path (str): Full path of the module.
        class_name (str): Class name to find.

    Returns:
        type: Type of specified class.
    """
    target_module = import_module(module_path)

    return getattr(target_module, class_name)


def copy_dict(target: dict, source: dict) -> None:
    """Copy values from source to target dict.

    Args:
        target (dict): Target dictionary to copy to.
        source (dict): Source dictionary to copy from.
    """
    for k, v in source.items():
        if type(v) != dict:
            target[k] = v
        else:
            if k not in target:
                target[k] = {}

            copy_dict(target[k], v)


class SupplyChainConfiguration:
    """Configuration of supply chain scenario."""

    def __init__(self) -> None:
        # Data model definitions.
        self.data_model_defs: Dict[str, DataModelDef] = {}

        # Entity (Unit & Facility) definitions.
        self.entity_defs: Dict[str, EntityDef] = {}

        # World configurations.
        self.world = {}

        # Other settings.
        self.settings = {}

        # Policy parameters, optional.
        self.policy_parameters = {}

    def add_data_definition(self, alias: str, class_name: str, module_path: str, name_in_frame: str) -> None:
        """Add a data model definition.

        Args:
            alias (str): Alias of this data model.
            class_name (str): Name of class.
            module_path (str): Full path of module.
            name_in_frame (str): Data model name in frame.
        """
        # Check conflicting.
        assert alias not in self.data_model_defs

        self.data_model_defs[alias] = DataModelDef(
            alias,
            module_path,
            class_name,
            find_class_type(module_path, class_name),
            name_in_frame,
        )

    def add_entity_definition(self, alias: str, class_name: str, module_path: str, data_model: Optional[str]) -> None:
        """Add entity (unit & facility) definition.

        Args:
            alias (str): Alias of this data model.
            class_name (str): Name of class.
            module_path (str): Full path of module.
            data_model (Optional[str]): Data model used for this entity. None indicates no corresponding data model.
        """
        assert alias not in self.entity_defs

        self.entity_defs[alias] = EntityDef(
            alias,
            module_path,
            class_name,
            find_class_type(module_path, class_name),
            data_model,
        )


class ConfigParser:
    """Supply chain configuration parser."""

    def __init__(self, core_path: str, config_path: str) -> None:
        self._result: Optional[SupplyChainConfiguration] = None

        self._core_path = core_path
        self._config_path = config_path

    def parse(self) -> SupplyChainConfiguration:
        """Parse configuration of current scenario.

        Returns:
            SupplyChainConfiguration: Configuration result of this scenario.
        """
        if self._result is None:
            self._result = SupplyChainConfiguration()
            self._parse_core()
            self._parse_config()

        return self._result

    def _parse_core(self) -> None:
        """Parse configuration from core.yml."""
        with open(self._core_path, "rt") as fp:
            conf = safe_load(fp)

            self._parse_core_conf(conf)

    def _parse_core_conf(self, conf: dict) -> None:
        # Data models
        if "datamodels" in conf:
            for module_conf in conf["datamodels"]["modules"]:
                module_path = module_conf["path"]

                for class_alias, class_def in module_conf["definitions"].items():
                    self._result.add_data_definition(
                        class_alias,
                        class_def["class"],
                        module_path,
                        class_def["name_in_frame"],
                    )

        # Entities
        for entity_type in ["units", "facilities"]:
            if entity_type in conf:
                for module_conf in conf[entity_type]["modules"]:
                    module_path = module_conf["path"]

                    for class_alias, class_def in module_conf["definitions"].items():
                        self._result.add_entity_definition(
                            class_alias,
                            class_def["class"],
                            module_path,
                            class_def["datamodel"],
                        )

    def _parse_config(self) -> None:
        """Parse configurations."""
        with open(os.path.join(self._config_path, "config.yml"), "rt", encoding="utf-8") as fp:
            conf = safe_load(fp)

            # Read customized core part.
            customized_core_conf = conf.get("core", None)

            if customized_core_conf is not None:
                self._parse_core_conf(customized_core_conf)  # TODO: there may be sth. wrong with the assert.

            # Facility definitions is not required, but it would be much simple to config with it
            facility_definitions = conf.get("facility_definitions", {})
            world_def = conf["world"]

            # Go through world configurations to generate a full one.
            # . Copy other configurations first
            for sub_conf_name in ("skus", "topology"):
                self._result.world[sub_conf_name] = world_def[sub_conf_name]

            # . Copy facilities content different if without definition reference.
            # or copy from definition first, then override with current.
            self._result.world["facilities"] = []

            for facility_conf in world_def["facilities"]:
                facility_ref = facility_conf.get("definition_ref", None)

                facility = {}

                if facility_ref is not None:
                    # Copy definition from base.
                    copy_dict(facility, facility_definitions[facility_ref])

                # Override with current.
                copy_dict(facility, facility_conf)

                self._result.world["facilities"].append(facility)

            self._result.settings = conf.get("settings", {})

            self._result.policy_parameters = conf.get("policy_parameters", {})
