
import numpy as np

from collections import defaultdict, namedtuple

from .frame_builder import build_frame

from tcod.path import AStar

from .config_parser import SupplyChainConfiguration

from .data import FacilityDataModel


# sku definition in world level
# bom is a dictionary, key is the material sku id, value is units per lot
Sku = namedtuple("Sku", ("name", "id", "bom", "output_units_per_lot"))


class World:
    def __init__(self):
        # all the facilities in this world, key: id, value: facilities
        self.facilities = {}

        # mapping from facility name to id
        self._facility_name2id_mapping = {}

        # all the entities (units and facilities) in this world
        # id -> instance
        self._entities = {}

        # frame of this world, this is determined by the unit selected.
        self.frame = None

        # id counter for all units and facilities in this world
        self._id_counter = 1

        # collection of data model class used in this world.
        self._data_class_collection = defaultdict(int)

        # sku collection of this world
        self._sku_collection = {}

        # sku id -> name in collection
        self._sku_id2name_mapping = {}

        # configuration of current world
        self.configs: dict = None

        # unit id to related data model index
        self.unit_id2index_mapping = {}

        # a star path finder
        self._path_finder: AStar = None

        self._data_model_definitions = None
        self._facility_definitions = None
        self._unit_definitions = None

    def gen_id(self):
        """Generate id for facility or unit."""
        new_id = self._id_counter

        self._id_counter += 1

        return new_id

    def build_unit(self, name: str):
        """Build an unit instance from it name via current configuration."""
        assert name in self._unit_definitions

        unit = self._unit_definitions[name].class_type()

        unit.id = self.gen_id()

        self._entities[unit.id] = unit

        return unit

    def build(self, all_in_one_config: SupplyChainConfiguration, snapshot_number: int):
        """Build current world according to configurations."""
        self.configs = all_in_one_config.world
        self._facility_definitions = all_in_one_config.facilities
        self._unit_definitions = all_in_one_config.units
        self._data_model_definitions = all_in_one_config.data_models

        configs = self.configs

        # collect sku information first
        for sku_conf in configs["skus"]:
            sku = Sku(sku_conf["name"], sku_conf["id"], {}, sku_conf["output_units_per_lot"])

            self._sku_id2name_mapping[sku.id] = sku.name
            self._sku_collection[sku.name] = sku

        # collect bom
        for sku_conf in configs["skus"]:
            sku = self._sku_collection[sku_conf["name"]]

            bom = sku_conf.get("bom", {})

            for material_sku_name, units_per_lot in bom.items():
                sku.bom[self._sku_collection[material_sku_name].id] = units_per_lot

        # build facilities first
        for facility_conf in configs["facilities"]:
            facility_name = facility_conf["name"]

            # create a new instance of facility
            facility_def = self._facility_definitions[facility_conf["class"]]
            facility = facility_def.class_type()

            # NOTE: DO set these fields before other operations.
            facility.world = self
            facility.id = self.gen_id()
            facility.name = facility_name

            self._facility_name2id_mapping[facility_name] = facility.id
            self.facilities[facility.id] = facility
            self._entities[facility.id] = facility

            # build the facility first to create related components.
            facility.build(facility_conf["configs"])

        # build the frame
        # . collect data model class
        data_class_in_frame = []

        for class_name, number in self._data_class_collection.items():
            class_def = self._data_model_definitions[class_name]

            data_class_in_frame.append((
                class_def.class_type,
                class_def.name_in_frame,
                number
            ))

        # add facility data model to frame
        data_class_in_frame.append((
            FacilityDataModel,
            "facilities",
            len(self.facilities),
        ))

        # . build the frame
        self.frame = build_frame(True, snapshot_number, data_class_in_frame)

        # construct the upstream topology
        topology = configs.get("topology", {})

        for cur_facility_name, topology_conf in topology.items():
            facility = self.get_facility_by_name(cur_facility_name)

            facility.upstreams = {}

            for sku_name, source_facilities in topology_conf.items():
                sku = self.get_sku(sku_name)

                facility.upstreams[sku.id] = [self.get_facility_by_name(source_name).id for source_name in source_facilities]

        # then initialize all facilities as we have the data instance.
        facility_node_index = 0

        for _, facility in self.facilities.items():
            facility.data = self.frame.facilities[facility_node_index]
            facility_node_index += 1

            facility.initialize()

        # construct the map grid
        grid_config = configs["grid"]

        grid_width, grid_height = grid_config["size"]

        # travel cost for a star path finder, 0 means block, > 1 means the cost travel to that cell
        # current all traversable cell's cost will be 1.
        cost_grid = np.ones(shape=(grid_width, grid_height), dtype=np.int8)

        # add blocks to grid
        for facility_name, facility_pos in grid_config["facilities"].items():
            facility_id = self._facility_name2id_mapping[facility_name]
            facility = self.facilities[facility_id]

            facility.x = facility_pos[0]
            facility.y = facility_pos[1]

            # facility cannot be a block, or we cannot find path to it,
            # but we can give it a big cost
            cost_grid[facility.x, facility.y] = 120

        for block_pos in grid_config["blocks"].values():
            cost_grid[block_pos[0], block_pos[1]] = 0

        # 0 for 2nd parameters means disable diagonal movement, so just up, right, down or left.
        self._path_finder = AStar(cost_grid, 0)

    def get_facility_by_id(self, facility_id: int):
        return self.facilities[facility_id]

    def get_facility_by_name(self, name: str):
        return self.facilities[self._facility_name2id_mapping[name]]

    def get_entity(self, entity_id: int):
        return self._entities[entity_id]

    def register_data_class(self, unit_id: int, name: str):
        assert name in self._data_model_definitions

        node_index = self._data_class_collection[name]

        self._data_class_collection[name] += 1
        self.unit_id2index_mapping[unit_id] = node_index

        return node_index

    def get_data_instance(self, class_name: str, node_index: int):
        alias = self._data_model_definitions[class_name].name_in_frame

        return getattr(self.frame, alias)[node_index]

    def get_sku(self, name: str):
        return self._sku_collection.get(name, None)

    def get_sku_by_id(self, sku_id: int):
        return self._sku_collection[self._sku_id2name_mapping[sku_id]]

    def find_path(self, start_x: int, start_y: int, goal_x: int, goal_y: int):
        return self._path_finder.get_path(int(start_x), int(start_y), int(goal_x), int(goal_y))

    def get_node_mapping(self):
        facility_info_dict = {facility_id: facility.get_node_info() for facility_id, facility in self.facilities.items()}

        # pick unit id and related index and node name
        id2index_mapping = {}

        for facility in facility_info_dict.values():
            for units in facility["units"].values():
                if type(units) is dict:
                    # one unit
                    id2index_mapping[units["id"]] = (units["node_name"], units["node_index"])
                elif type(units) is list:
                    for unit in units:
                        id2index_mapping[unit["id"]] = (unit["node_name"], unit["node_index"])

        return {
            "mapping": id2index_mapping,
            "detail": facility_info_dict
        }

    def _build_facility(self, conf: dict):
        name = conf["name"]
        class_alias = conf["class"]

        facility_def = self.configs.facilities[class_alias]

        facility = facility_def.class_type()

        facility.id = self.gen_id()
        facility.world = self
        facility.name = name

        facility.build(conf["configs"])

        return facility
