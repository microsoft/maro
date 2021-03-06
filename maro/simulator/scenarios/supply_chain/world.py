
import numpy as np

from collections import defaultdict, namedtuple

from .frame_builder import build_frame

from .configs import unit_mapping, data_class_mapping

from tcod.path import AStar


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

    def gen_id(self):
        """Generate id for facility or unit."""
        new_id = self._id_counter

        self._id_counter += 1

        return new_id

    def build_unit(self, name: str):
        """Build an unit instance from it name via current configuration."""
        assert name in unit_mapping

        unit = unit_mapping[name]["class"]()

        unit.id = self.gen_id()

        self._entities[unit.id] = unit

        return unit

    def build(self, configs: dict, snapshot_number: int):
        """Build current world according to configurations."""
        self.configs = configs

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
            facility = facility_conf["class"]()

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
            class_config = data_class_mapping[class_name]

            data_class_in_frame.append((
                class_config["class"],
                class_config["alias_in_snapshot"],
                number
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
        for _, facility in self.facilities.items():
            facility.initialize()

        # construct the map grid
        grid_config = configs["grid"]

        grid_width, grid_height = grid_config["size"]

        # travel cost for a star path finder, 0 means block, > 1 means the cost travel to that cell
        # current all traversable cell's cost will be 1.
        cost_grid = np.ones(shape=(grid_width, grid_height), dtype=np.int)

        # add blocks to grid
        for facility_pos in grid_config["facilities"].values():
            cost_grid[facility_pos[0], facility_pos[1]] = 0

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
        assert name in data_class_mapping

        node_index = self._data_class_collection[name]

        self._data_class_collection[name] += 1
        self.unit_id2index_mapping[unit_id] = node_index

        return node_index

    def get_data_instance(self, class_name: str, node_index: int):
        alias = data_class_mapping[class_name]["alias_in_snapshot"]

        return getattr(self.frame, alias)[node_index]

    def get_sku(self, name: str):
        return self._sku_collection.get(name, None)

    def get_sku_by_id(self, sku_id: int):
        return self._sku_collection[self._sku_id2name_mapping[sku_id]]

    def find_path(self, start_x: int, start_y: int, goal_x: int, goal_y: int):
        return self._path_finder.get_path(start_x, start_y, goal_x, goal_y)

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
