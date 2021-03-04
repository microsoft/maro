
from collections import defaultdict, namedtuple

from .frame_builder import build_frame

from .configs import unit_mapping, data_class_mapping


# sku definition in world level
Sku = namedtuple("Sku", ("name", "id"))


class World:
    def __init__(self):
        # all the facilities in this world, key: id, value: facilities
        self.facilities = {}

        # frame of this world, this is determined by the unit selected.
        self.frame = None

        # id counter for all units and facilities in this world
        self._id_counter = 1

        # collection of data model class used in this world.
        self._data_class_collection = defaultdict(int)

        # sku collection of this world
        self._sku_collection = {}

        # configuration of current world
        self.configs: dict = None

    def gen_id(self):
        """Generate id for facility or unit."""
        new_id = self._id_counter

        self._id_counter += 1

        return new_id

    def build_unit(self, name: str):
        """Build an unit instance from it name via current configuration."""
        assert name in unit_mapping

        logic = unit_mapping[name]["class"]()

        logic.id = self.gen_id()

        return logic

    def build(self, configs: dict, snapshot_number: int):
        """Build current world according to configurations."""
        self.configs = configs

        # collect sku information first
        for sku_conf in configs["skus"]:
            sku = Sku(sku_conf["name"], sku_conf["id"])

            self._sku_collection[sku.name] = sku

        # build facilities first
        for facility_name, facility_conf in configs["facilities"].items():
            # create a new instance of facility
            facility = facility_conf["class"]()

            # NOTE: DO set these fields before other operations.
            facility.world = self
            facility.id = self.gen_id()

            self.facilities[facility_name] = facility

            # build the facility first to create related components.
            self.facilities[facility_name].build(facility_conf["configs"])

        # and build the frame
        self.frame = build_frame(
            True,
            snapshot_number,
            [(data_class_mapping[class_name]["class"], data_class_mapping[class_name]["alias_in_snapshot"], number) for class_name, number in self._data_class_collection.items()])

        # then initialize all facilities as we have the data instance.
        for _, facility in self.facilities.items():
            facility.initialize()

    def register_data_class(self, name: str):
        assert name in data_class_mapping

        node_index = self._data_class_collection[name]

        self._data_class_collection[name] += 1

        return node_index

    def get_data_instance(self, class_name: str, node_index: int):
        alias = data_class_mapping[class_name]["alias_in_snapshot"]

        return getattr(self.frame, alias)[node_index]

    def get_sku(self, name: str):
        return self._sku_collection.get(name, None)
