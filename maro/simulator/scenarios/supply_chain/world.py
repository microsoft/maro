
from collections import defaultdict

from .frame_builder import build_frame

from .configs import logic_mapping, datamodel_mapping


class World:
    def __init__(self):
        self.facilities = {}
        self.frame = None

        self._id_counter = 1
        self._datamodel_collection = defaultdict(int)

    def build_logic(self, name: str):
        assert name in logic_mapping

        logic = logic_mapping[name]["class"]()

        logic.id = self._id_counter

        self._id_counter += 1

        return logic

    def build(self, configs: dict):
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
            10,
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
