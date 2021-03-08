
from importlib import import_module

from yaml import safe_load

from collections import namedtuple


DataModelItem = namedtuple("DataModelItem", ("alias", "module_path", "class_name", "class_type"))
UnitItem = namedtuple("UnitItem", ("alias", "module_path", "class_name", "class_type", "configs"))
FacilityItem = namedtuple("FacilityItem", ("alias", "module_path", "class_name", "class_type", "configs"))


def find_class_type(module_path: str, class_name: str):
    target_module = import_module(module_path)

    return getattr(target_module, class_name)


class SupplyChainConfiguration:
    data_models = None
    units = None
    facilities = None
    world = None

    def __init__(self):
        self.data_models = {}
        self.units = {}
        self.facilities = {}
        self.world = {}

    def add_data_definition(self, alias: str, class_name: str, module_path: str):
        # check conflict
        assert alias not in self.data_models

        self.data_models[alias] = DataModelItem(alias, module_path, class_name, find_class_type(module_path, class_name))

    def add_unit_definition(self, alias: str, class_name: str, module_path: str, configs: dict):
        assert alias not in self.units

        self.units[alias] = UnitItem(alias, module_path, class_name, find_class_type(module_path, class_name), configs)

    def add_facility_definition(self, alias: str, class_name: str, module_path: str, configs: dict):
        assert alias not in self.facilities

        self.facilities[alias] = FacilityItem(alias, module_path, class_name, find_class_type(module_path, class_name), configs)



class ConfigParser:
    def __init__(self, core_file: str, config_file: str):
        self._core_file = core_file
        self._config_file = config_file

        self._result = SupplyChainConfiguration()

    def parse(self):
        self._parse_core()
        self._parse_world_config()

        return self._result

    def _parse_core(self):
        with open(self._core_file, "rt") as fp:
            core_config = safe_load(fp)

            self._read_core_conf(core_config)

    def _read_core_conf(self, core_config: dict):
        # data models
        if "data" in core_config:
            for module_conf in core_config["data"]["modules"]:
                module_path = module_conf["path"]

                for class_alias, class_def in module_conf["definitions"].items():
                    self._result.add_data_definition(class_alias, class_def["class"], module_path)

        # units
        if "units" in core_config:
            for module_conf in core_config["units"]["modules"]:
                module_path = module_conf["path"]

                for class_alias, class_def in module_conf["definitions"].items():
                    self._result.add_unit_definition(class_alias, class_def["class"], module_path, class_def)

        # facilities
        if "facilities" in core_config:
            for module_conf in core_config["facilities"]["modules"]:
                module_path = module_conf["path"]

                for class_alias, class_def in module_conf["definitions"].items():
                    self._result.add_facility_definition(class_alias, class_def["class"], module_path, class_def)

    def _parse_world_config(self):
        with open(self._config_file, "rt") as fp:
            world_conf = safe_load(fp)

            # read and override core part
            customized_core_conf = world_conf.get("core", {})

            self._read_core_conf(customized_core_conf)

            # read the world config first
            self._result.world = world_conf["world"]

            # then try to fulfill with core configurations
            for facility_conf in self._result.world["facilities"]:
                facility_class_alias = facility_conf["class"]

                facility_def = self._result.facilities[facility_class_alias]

                for property_name, property_conf in facility_def.configs.items():
                    if property_name == "class":
                        continue

                    if property_name not in facility_conf:
                        facility_conf["configs"][property_name] = property_conf
                    else:
                        pass


if __name__ == "__main__":
    parser = ConfigParser("maro/simulator/scenarios/supply_chain/topologies/core.yaml", "maro/simulator/scenarios/supply_chain/topologies/sample1/config.yml")

    result = parser.parse()

    print(result.world)
