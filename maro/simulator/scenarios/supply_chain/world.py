# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from typing import List, NamedTuple, Optional, Tuple, Union

import networkx as nx
from maro.backends.frame import FrameBase

from .easy_config import EasyConfig, SkuInfo
from .facilities import FacilityBase
from .frame_builder import build_frame
from .parser import DataModelDef, FacilityDef, SupplyChainConfiguration, UnitDef
from .units import ExtendUnitBase, UnitBase


class AgentInfo(NamedTuple):
    id: int
    agent_type: object
    is_facility: bool
    sku: Optional[SkuInfo]
    facility_id: int
    parent_id: Optional[int]


class World:
    """Supply chain world contains facilities and grid base map."""

    def __init__(self):
        # Frame for current world configuration.
        self.frame: Optional[FrameBase] = None

        # Current configuration.
        self.configs: Optional[SupplyChainConfiguration] = None

        # Durations of current simulation.
        self.durations = 0

        # All the entities in the world.
        self.units = {}

        # All the facilities in this world.
        self.facilities = {}

        # Entity id counter, every unit and facility have unique id.
        self._id_counter = 1

        # Grid of the world
        self._graph: Optional[nx.Graph] = None

        # Sku id to name mapping, used for querying.
        self._sku_id2name_mapping = {}

        # All the sku in this world.
        self._sku_collection = {}

        # Facility name to id mapping, used for querying.
        self._facility_name2id_mapping = {}

        # Data model class collection, used to collection data model class and their number in frame.
        self._data_class_collection = {}

        self.agent_list = []

        self.agent_type_dict = {}

        self.max_sources_per_facility = 0
        self.max_price = 0

    def get_sku_by_name(self, name: str) -> EasyConfig:
        """Get sku information by name.

        Args:
            name (str): Sku name to query.

        Returns:
            EasyConfig: General information for sku, used as a dict, but support use key as property.
        """
        return self._sku_collection.get(name, None)

    def get_sku_by_id(self, sku_id: int) -> EasyConfig:
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

    def get_entity(self, id: int) -> Union[FacilityBase, UnitBase]:
        """Get an entity(Unit or Facility) by id.

        Args:
            id (int): Id to query.

        Returns:
            Union[FacilityBase, UnitBase]: Unit or facility instance.
        """
        if id in self.units:
            return self.units[id]
        else:
            return self.facilities[id]

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
        return nx.astar_path(self._graph, source=(start_x, start_y), target=(goal_x, goal_y), weight="cost")

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
            sku = SkuInfo(sku_conf)

            self._sku_id2name_mapping[sku.id] = sku.name
            self._sku_collection[sku.name] = sku

        # Collect bom info.
        for sku_conf in world_config["skus"]:
            sku = self._sku_collection[sku_conf["name"]]
            sku.bom = {}

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

            # Due with data model.
            data_model_def: DataModelDef = self.configs.data_models[facility_def.data_model_alias]

            # Register the data model, so that it will help to generate related instance index.
            facility.data_model_index = self._register_data_model(data_model_def.alias)
            facility.data_model_name = data_model_def.name_in_frame

            # Build children (units).
            for child_name, child_conf in facility_conf["children"].items():
                setattr(facility, child_name, self.build_unit(facility, facility, child_conf))

            self.facilities[facility.id] = facility

            self._facility_name2id_mapping[facility.name] = facility.id

        # Build frame.
        self.frame = self._build_frame(snapshot_number)

        # Assign data model instance.
        for unit in self.units.values():
            if unit.data_model_name is not None:
                unit.data_model = getattr(self.frame, unit.data_model_name)[unit.data_model_index]

        for facility in self.facilities.values():
            if facility.data_model_name is not None:
                facility.data_model = getattr(self.frame, facility.data_model_name)[facility.data_model_index]

        # Construct the upstream topology.
        topology = world_config["topology"]

        for cur_facility_name, topology_conf in topology.items():
            facility = self.get_facility_by_name(cur_facility_name)

            for sku_name, source_facilities in topology_conf.items():
                sku = self.get_sku_by_name(sku_name)
                facility.upstreams[sku.id] = []

                self.max_sources_per_facility = max(self.max_sources_per_facility, len(source_facilities))

                for source_name in source_facilities:
                    source_facility = self.get_facility_by_name(source_name)

                    facility.upstreams[sku.id].append(source_facility)

                    if sku.id not in source_facility.downstreams:
                        source_facility.downstreams[sku.id] = []

                    source_facility.downstreams[sku.id].append(facility)

        # Call initialize method for facilities.
        for facility in self.facilities.values():
            facility.initialize()

        # Call initialize method for units.
        for unit in self.units.values():
            unit.initialize()

        # Construct the map grid.
        grid_config = world_config["grid"]

        grid_width, grid_height = grid_config["size"]

        # Build our graph base one settings.
        # This will create a full connect graph.
        self._graph = nx.grid_2d_graph(grid_width, grid_height)

        # All edge weight will be 1 by default.
        edge_weights = {e: 1 for e in self._graph.edges()}

        # Facility to cell will have 1 weight, cell to facility will have 4 cost.
        for facility_name, pos in grid_config["facilities"].items():
            facility_id = self._facility_name2id_mapping[facility_name]
            facility = self.facilities[facility_id]
            facility.x = pos[0]
            facility.y = pos[1]
            pos = tuple(pos)

            # Neighbors to facility will have high cost.
            for npos in ((pos[0] - 1, pos[1]), (pos[0] + 1, pos[1]), (pos[0], pos[1] - 1), (pos[0], pos[1] + 1)):
                if 0 <= npos[0] < grid_width and 0 <= npos[1] < grid_height:
                    edge_weights[(npos, pos)] = 4

        nx.set_edge_attributes(self._graph, edge_weights, "cost")

        # Collection agent list
        for facility in self.facilities.values():
            agent_type = facility.configs.get("agent_type", None)

            if agent_type is not None:
                self.agent_list.append(AgentInfo(facility.id, agent_type, True, None, facility.id, None))

                self.agent_type_dict[type(facility).__name__] = agent_type

                for sku in facility.skus.values():
                    self.max_price = max(self.max_price, sku.price)

        # Find units that contains agent type to expose as agent.
        for unit in self.units.values():
            agent_type = unit.config.get("agent_type", None)

            if agent_type is not None:
                # Unit or facility id, agent type, is facility, sku info, facility id, parent id.
                self.agent_list.append(
                    AgentInfo(
                        unit.id,
                        agent_type,
                        False,
                        unit.facility.skus[unit.product_id],
                        unit.facility.id,
                        unit.parent.id
                    )
                )

                self.agent_type_dict[type(unit).__name__] = agent_type

    def build_unit_by_type(self, unit_def: UnitDef, parent: Union[FacilityBase, UnitBase], facility: FacilityBase):
        """Build an unit by its type.

        Args:
            unit_def (UnitDef): Definition of this unit.
            parent (Union[FacilityBase, UnitBase]): Parent of this unit.
            facility (FacilityBase): Facility this unit belongs to.

        Returns:
            UnitBase: Unit instance.
        """
        unit = unit_def.class_type()

        unit.id = self._gen_id()
        unit.parent = parent
        unit.facility = facility
        unit.world = self

        if unit_def.data_model_alias is not None:
            # Due with data model.
            data_model_def: DataModelDef = self.configs.data_models[unit_def.data_model_alias]

            # Register the data model, so that it will help to generate related instance index.
            unit.data_model_index = self._register_data_model(data_model_def.alias)
            unit.data_model_name = data_model_def.name_in_frame

        self.units[unit.id] = unit

        return unit

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
            self.units[unit_instance.id] = unit_instance

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
                unit_instance.children = []

                for child_name, child_conf in children_conf.items():
                    # If child configuration is a dict, then we add it as a property by name (key).
                    if type(child_conf) == dict:
                        child_instance = self.build_unit(facility, unit_instance, child_conf)

                        setattr(unit_instance, child_name, child_instance)
                        unit_instance.children.append(child_instance)
                    elif type(child_conf) == list:
                        # If child configuration is a list, then will treat it as list property, named same as key.
                        child_list = []
                        for conf in child_conf:
                            child_list.append(self.build_unit(facility, unit_instance, conf))

                        setattr(unit_instance, child_name, child_list)
                        unit_instance.children.extend(child_list)

            return unit_instance
        else:
            # If this is template unit, then will use the class' static method 'generate' to generate sub-units.
            children = unit_def.class_type.generate(facility, config.get("config"), unit_def)

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

        for unit_id, unit in self.units.items():
            sku = None

            if isinstance(unit, ExtendUnitBase):
                sku = unit.facility.skus[unit.product_id]

            if unit.data_model is not None:
                id2index_mapping[unit_id] = (unit.data_model_name, unit.data_model_index, unit.facility.id, sku)
            else:
                id2index_mapping[unit_id] = (None, None, unit.facility.id, sku)

        return {
            "agent_types": {k: v for k, v in self.agent_type_dict.items()},
            "unit_mapping": id2index_mapping,
            "skus": {sku.id: sku for sku in self._sku_collection.values()},
            "facilities": facility_info_dict,
            "max_price": self.max_price,
            "max_sources_per_facility": self.max_sources_per_facility,
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
