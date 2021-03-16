# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from collections import namedtuple
from typing import List, Union, Tuple

import numpy as np
from maro.backends.frame import FrameBase
from tcod.path import AStar

from .facilities import FacilityBase
from .frame_builder import build_frame
from .parser import SupplyChainConfiguration, DataModelDef, UnitDef, FacilityDef
from .units import UnitBase

SkuInfo = namedtuple("SkuInfo", ("name", "id", "bom", "output_units_per_lot"))


class World:
    """Supply chain world contains facilities and grid base map."""

    def __init__(self):
        # Frame for current world configuration.
        self.frame: FrameBase = None

        # Current configuration.
        self.configs: SupplyChainConfiguration = None

        # Durations of current simulation.
        self.durations = 0

        # All the entities in the world, include unit and facility.
        self.entities = {}

        # All the facilities in this world.
        self.facilities = {}

        # Entity id counter, every unit and facility have unique id.
        self._id_counter = 1

        # Path finder for production transport.
        self._path_finder: AStar = None

        # Sku id to name mapping, used for querying.
        self._sku_id2name_mapping = {}

        # All the sku in this world.
        self._sku_collection = {}

        # Facility name to id mapping, used for querying.
        self._facility_name2id_mapping = {}

        # Data model class collection, used to collection data model class and their number in frame.
        self._data_class_collection = {}

    def get_sku_by_name(self, name: str) -> SkuInfo:
        """Get sku information by name.

        Args:
            name (str): Sku name to query.

        Returns:
            SkuInfo: General information for sku.
        """
        return self._sku_collection.get(name, None)

    def get_sku_by_id(self, sku_id: int) -> SkuInfo:
        """Get sku information by sku id.

        Args:
            sku_id (int): Id of sku to query.

        Returns:
            SkuInfo: General information for sku.
        """
        return self._sku_collection[self._sku_id2name_mapping[sku_id]]

    def get_facility_by_id(self, facility_id: int) -> FacilityBase:
        """Get facility by id.

        Args:
            facility_id (int): Facility id to query.

        Returns:
            FacilityBase: Facility instance.
        """
        return self.facilities[facility_id]

    def get_facility_by_name(self, name: str):
        """Get facility by name.

        Args:
            name (str): Facility name to query.

        Returns:
            FacilityBase: Facility instance.
        """
        return self.facilities[self._facility_name2id_mapping[name]]

    def get_entity(self, entity_id: int) -> Union[FacilityBase, UnitBase]:
        """Get an entity (facility or unit) by id.

        Args:
            entity_id (int): Id to query.

        Returns:
            Union[FacilityBase, UnitBase]: Entity instance.
        """
        return self._entities[entity_id]

    def find_path(self, start_x: int, start_y: int, goal_x: int, goal_y: int) -> List[Tuple[int, int]]:
        """Find path to specified cell.

        Args:
            start_x (int): Start cell position x.
            start_y (int): Start cell position y.
            goal_x (int): Destination cell position x.
            goal_y (int): Destination cell position y.

        Returns:
            List[Tuple[int, int]]: List of (x, y) position to target.
        """
        return self._path_finder.get_path(int(start_x), int(start_y), int(goal_x), int(goal_y))

    def build(self, configs: SupplyChainConfiguration, snapshot_number: int, durations: int):
        """Build world with configurations.

        Args:
            configs (SupplyChainConfiguration): Configuration of current world.
            snapshot_number (int): Number of snapshots to keep in memory.
            durations (int): Durations of current simulation.
        """
        self.durations = durations
        self.configs = configs

        world_config = configs.world

        # Grab sku information for this world.
        for sku_conf in world_config["skus"]:
            sku = SkuInfo(sku_conf["name"], sku_conf["id"], {}, sku_conf["output_units_per_lot"])

            self._sku_id2name_mapping[sku.id] = sku.name
            self._sku_collection[sku.name] = sku

        # Collect bom info.
        for sku_conf in world_config["skus"]:
            sku = self._sku_collection[sku_conf["name"]]

            bom = sku_conf.get("bom", {})

            for material_sku_name, units_per_lot in bom.items():
                sku.bom[self._sku_collection[material_sku_name].id] = units_per_lot

        # Construct facilities.
        for facility_conf in world_config["facilities"]:
            facility_class_alias = facility_conf["class"]
            facility_def: FacilityDef = self.configs.facilities[facility_class_alias]
            facility_class_type = facility_def.class_type

            # Instance of facility.
            facility = facility_class_type()

            # Normal properties.
            facility.id = self._gen_id()
            facility.name = facility_conf["name"]
            facility.world = self

            # Parse sku info.
            facility.parse_skus(facility_conf["skus"])

            # Parse config for facility.
            facility.parse_configs(facility_conf.get("config", {}))

            # Build children (units).
            for child_name, child_conf in facility_conf["children"].items():
                setattr(facility, child_name, self.build_unit(facility, facility, child_conf))

            self.facilities[facility.id] = facility

            self._facility_name2id_mapping[facility.name] = facility.id

        # Build frame.
        self.frame = self._build_frame(snapshot_number)

        # Assign data model instance.
        for agent in self.entities.values():
            if agent.data_model_name is not None:
                agent.data_model = getattr(self.frame, agent.data_model_name)[agent.data_model_index]

        # Construct the upstream topology.
        topology = world_config["topology"]

        for cur_facility_name, topology_conf in topology.items():
            facility = self.get_facility_by_name(cur_facility_name)

            facility.upstreams = {}

            for sku_name, source_facilities in topology_conf.items():
                sku = self.get_sku_by_name(sku_name)

                facility.upstreams[sku.id] = [
                    self.get_facility_by_name(source_name).id for source_name in source_facilities
                ]

        # Call initialize method for facilities.
        for facility in self.facilities.values():
            facility.initialize()

        # Call initialize method for units.
        for agent in self.entities.values():
            agent.initialize()

        # TODO: replace tcod with other lib.
        # Construct the map grid.
        grid_config = world_config["grid"]

        grid_width, grid_height = grid_config["size"]

        # Travel cost for a star path finder, 0 means block, > 1 means the cost travel to that cell
        # current all traversable cell's cost will be 1.
        cost_grid = np.ones(shape=(grid_width, grid_height), dtype=np.int8)

        # Add blocks to grid.
        for facility_name, facility_pos in grid_config["facilities"].items():
            facility_id = self._facility_name2id_mapping[facility_name]
            facility = self.facilities[facility_id]

            facility.x = facility_pos[0]
            facility.y = facility_pos[1]

            # Facility cannot be a block, or we cannot find path to it,
            # but we can give it a big cost
            cost_grid[facility.x, facility.y] = 120

        for block_pos in grid_config["blocks"].values():
            cost_grid[block_pos[0], block_pos[1]] = 0

        # 0 for 2nd parameters means disable diagonal movement, so just up, right, down or left.
        self._path_finder = AStar(cost_grid, 0)

    def build_unit(self, facility: FacilityBase, parent: Union[FacilityBase, UnitBase], config: dict) -> UnitBase:
        """Build an unit by its configuration.

        Args:
            facility (FacilityBase): Facility of this unit belongs to.
            parent (Union[FacilityBase, UnitBase]): Parent of this unit belongs to, this may be same with facility, if
                this unit is attached to a facility.
            config (dict): Configuration of this unit.

        Returns:
            UnitBase: Unit instance.
        """
        unit_class_alias = config["class"]
        unit_def: UnitDef = self.configs.units[unit_class_alias]

        is_template = config.get("is_template", False)

        # If it is not a template, then just use current configuration to generate unit.
        if not is_template:
            unit_instance = unit_def.class_type()

            # Assign normal properties.
            unit_instance.id = self._gen_id()
            unit_instance.world = self
            unit_instance.facility = facility
            unit_instance.parent = parent

            # Record the id.
            self.entities[unit_instance.id] = unit_instance

            # Due with data model.
            data_model_def: DataModelDef = self.configs.data_models[unit_def.data_model_alias]

            # Register the data model, so that it will help to generate related instance index.
            unit_instance.data_model_index = self._register_data_model(data_model_def.alias)
            unit_instance.data_model_name = data_model_def.name_in_frame

            # Parse the config is there is any.
            unit_instance.parse_configs(config.get("config", {}))

            # Prepare children.
            children_conf = config.get("children", None)

            if children_conf:
                for child_name, child_conf in children_conf.items():
                    # If child configuration is a dict, then we add it as a property by name (key).
                    if type(child_conf) == dict:
                        setattr(unit_instance, child_name, self.build_unit(facility, unit_instance, child_conf))
                    elif type(child_conf) == list:
                        # If child configuration is a list, then will treat it as list property, named same as key.
                        child_list = []
                        for conf in child_conf:
                            child_list.append(self.build_unit(facility, unit_instance, conf))

                        setattr(unit_instance, child_name, child_list)

            return unit_instance
        else:
            # If this is template unit, then will use the class' static method 'generate' to generate sub-units.
            children = unit_def.class_type.generate(facility, config.get("config"))

            for child in children.values():
                child.id = self._gen_id()
                child.world = self
                child.facility = facility
                child.parent = parent

                # Pass the config if there is any.
                child.parse_configs(config.get("config", {}))

                self.entities[child.id] = child

            return children

    def get_node_mapping(self):
        """Collect all the entities information.

        Returns:
            dict: A dictionary contains 'mapping' for id to data model index mapping,
                'detail' for detail of units and facilities.
        """
        facility_info_dict = {
            facility_id: facility.get_node_info() for facility_id, facility in self.facilities.items()
        }

        id2index_mapping = {}

        for facility in facility_info_dict.values():
            for units in facility["units"].values():
                if type(units) is dict:
                    id2index_mapping[units["id"]] = (units["node_name"], units["node_index"])
                elif type(units) is list:
                    for unit in units:
                        id2index_mapping[unit["id"]] = (unit["node_name"], unit["node_index"])

        return {
            "mapping": id2index_mapping,
            "detail": facility_info_dict
        }

    def _register_data_model(self, alias: str) -> int:
        """Register a data model alias, used to collect data model used in frame.

        Args:
            alias (str): Class alias defined in core.yml.

        Returns:
            int: Specified data model instance index after frame is built.
        """
        if alias not in self._data_class_collection:
            self._data_class_collection[alias] = 0

        node_index = self._data_class_collection[alias]

        self._data_class_collection[alias] += 1

        return node_index

    def _build_frame(self, snapshot_number: int) -> FrameBase:
        """Build frame by current world definitions.

        Args:
            snapshot_number (int): Number of snapshots to keep in memory.

        Returns:
            FrameBase: Frame instance with data model in current configuration.
        """
        data_class_in_frame = []

        for alias, number in self._data_class_collection.items():
            data_model_def: DataModelDef = self.configs.data_models[alias]
            data_class_in_frame.append((
                data_model_def.class_type,
                data_model_def.name_in_frame,
                number
            ))

        frame = build_frame(True, snapshot_number, data_class_in_frame)

        return frame

    def _gen_id(self):
        """Generate id for entities."""
        nid = self._id_counter

        self._id_counter += 1

        return nid
