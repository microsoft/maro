
from collections import defaultdict, namedtuple

from .frame_builder import build_frame

from .configs import unit_mapping, datamodel_mapping


Sku = namedtuple("Sku", ("name", "id"))


class World:
    def __init__(self):
        self.facilities = {}
        self.frame = None

        self._id_counter = 1
        self._datamodel_collection = defaultdict(int)
        self._sku_collection = {}

    def build_unit(self, name: str):
        assert name in unit_mapping

        logic = unit_mapping[name]["class"]()

        logic.id = self._id_counter

        self._id_counter += 1

        return logic

    def build(self, configs: dict, snapshot_number: int):
        # collect sku information first
        for sku_conf in configs["skus"]:
            sku = Sku(sku_conf["name"], sku_conf["id"])

            self._sku_collection[sku.name] = sku

        # build facilities first
        for facility_name, facility_conf in configs["facilities"].items():
            # create a new instance of facility
            facility = facility_conf["class"]()

            facility.world = self
            facility.id = self._id_counter

            self._id_counter += 1

            self.facilities[facility_name] = facility

            self.facilities[facility_name].build(facility_conf["configs"])

        # and build the frame
        self.frame = build_frame(
            True,
            snapshot_number,
            [(datamodel_mapping[class_name]["class"], datamodel_mapping[class_name]["alias_in_snapshot"], number) for class_name, number in self._datamodel_collection.items()])

        # then initialize all facilities
        for _, facility in self.facilities.items():
            facility.initialize()

    def register_datamodel(self, name: str):
        assert name in datamodel_mapping

        node_index = self._datamodel_collection[name]

        self._datamodel_collection[name] += 1

        return node_index

    def get_datamodel(self, class_name: str, node_index: int):
        alias = datamodel_mapping[class_name]["alias_in_snapshot"]

        return getattr(self.frame, alias)[node_index]

    def get_sku(self, name: str):
        return self._sku_collection.get(name, None)
