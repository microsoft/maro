# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import tcod
import pprint

from tabulate import tabulate
from maro.simulator import Env
from maro.simulator.scenarios.supply_chain import ConsumerAction
from state_shaping import SupplyChainStateShaping


WINDOW_SCALE_FACTOR = 2
TILESET_SIZE = 8

CHAR_DEFAULT = 0x2610


# https://stackoverflow.com/questions/287871/how-to-print-colored-text-to-the-terminal
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class InteractiveRenderaleEnv:
    def __init__(self, env: Env):
        self.env = env

        self._state_shaping = SupplyChainStateShaping(env)

    def start(self):
        print(f"{bcolors.WARNING}Press SPACE for next step!!!{bcolors.ENDC}")
        print(f"{bcolors.WARNING}Press S to show current debug states!!!{bcolors.ENDC}")
        print(f"{bcolors.WARNING}Press M to show env summary!!!{bcolors.ENDC}")
        print(f"{bcolors.WARNING}Press A to choose action, {bcolors.FAIL}NOTE: action must be none at beginning!!!{bcolors.ENDC}")
        print(f"{bcolors.WARNING}Press ESCAPE to clear action!!!{bcolors.ENDC}")

        # tileset from https://github.com/libtcod/python-tcod.
        tileset = tcod.tileset.load_tilesheet("font/dejavu10x10_gs_tc.png", 32, TILESET_SIZE, tcod.tileset.CHARMAP_TCOD)

        grid_width, grid_height = self.env.configs["grid"]["size"]

        # blocks
        facilities = []
        railroads = []

        for facility, pos in self.env.configs["grid"]["facilities"].items():
            facilities.append(pos)

        for pos_list in self.env.configs["grid"]["blocks"].values():
            railroads.extend(pos_list)

        console = tcod.Console(grid_width, grid_height)

        with tcod.context.new(
            width=grid_width * TILESET_SIZE * WINDOW_SCALE_FACTOR,
            height=grid_height * TILESET_SIZE * WINDOW_SCALE_FACTOR,
            renderer=tcod.RENDERER_SDL2,
            tileset=tileset,
            title="Supply chain environment"
        ) as ctx:
            action = None
            is_new_step = False

            while True:
                # clear
                console.clear()

                # draw background
                console.draw_rect(0, 0, 20, 20, CHAR_DEFAULT)

                # show facilities
                for facility in facilities:
                    console.print(facility[0], facility[1], "F", (0, 255, 0))

                for railroad in railroads:
                    console.print(railroad[0], railroad[1], "R", (255, 0, 0))

                # show vehicles that on the way
                self.present_vehicles(console)

                ctx.present(
                    console,
                    keep_aspect=True,
                    integer_scaling=True
                )

                for event in tcod.event.wait():
                    ctx.convert_event(event)

                    if event.type == "QUIT":
                        return

                    if event.type == "KEYDOWN":
                        # press SPACE to next step
                        if event.sym == tcod.event.K_SPACE:
                            # push environment to next step

                            metrics, decision_event, is_done = self.env.step(action)

                            is_new_step = True

                            # self.state_shaping()

                            print(f"{bcolors.OKGREEN}Current environment tick:", self.env.tick, f"{bcolors.ENDC}")

                            if is_done:
                                return
                        elif event.sym == tcod.event.K_s:
                            # show state we predefined
                            self.show_states()
                        elif event.sym == tcod.event.K_m:
                            # show summary
                            self.show_summary()
                        elif event.sym == tcod.event.K_a:
                            action = self.choose_action()
                        elif event.sym == tcod.event.K_ESCAPE:
                            action = None

    def present_vehicles(self, console: tcod.Console):
        vehicles = self.env.snapshot_list["transport"]
        vehicle_number = len(vehicles)

        # here we query the attributes that slot number ==1,
        # then query position, or snapshot_list will try to padding for id
        normal_list = vehicles[self.env.frame_index::("id", "steps", "location")].flatten().reshape(vehicle_number, -1).astype(np.int)
        pos_list = vehicles[self.env.frame_index::"position"].flatten().reshape(vehicle_number, -1).astype(np.int)

        for index, state in enumerate(normal_list):
            location = state[2]
            steps = state[1]

            if steps > 0:
                x, y = pos_list[index]

                if x >= 0 and y >= 0:
                    console.print(x, y, "V", (0, 255, 0), (128, 128, index))

    def state_shaping(self):
        print(self._state_shaping.shape())

    def show_summary(self):
        pp = pprint.PrettyPrinter(indent=2, depth=8)

        print(f"{bcolors.HEADER}env summary:{bcolors.ENDC}")

        pp.pprint(self.env.summary)

    def show_states(self):
        self.show_manufacture_states()

        self.show_vehicle_states()

        self.show_demand_states()

        self.show_storage_states()

    def show_storage_states(self):
        storages = self.env.snapshot_list["storage"]
        storage_number = len(storages)

        storage_features = ("id", "facility_id", "remaining_space", "capacity", "unit_storage_cost")
        storage_states = storages[self.env.frame_index::storage_features].flatten().reshape(storage_number, -1).astype(np.int)

        storage_all_states = []

        for index, state in enumerate(storage_states):
            product_list = storages[self.env.frame_index:index:"product_list"].flatten().astype(np.int)
            product_number = storages[self.env.frame_index:index:"product_number"].flatten().astype(np.int)

            storage_all_states.append(list(state) + [product_list, product_number])

        print(f"{bcolors.HEADER}Storage states:{bcolors.ENDC}")
        print(tabulate(storage_all_states, storage_features + ("product_list", "product_number")))

    def show_demand_states(self):
        sellers = self.env.snapshot_list["seller"]
        seller_number = len(sellers)

        seller_features = ("id", "facility_id", "product_id", "demand", "sold", "total_sold", "sale_gamma")
        seller_states = sellers[self.env.frame_index::seller_features].flatten().reshape(seller_number, -1).astype(np.int)

        print(f"{bcolors.HEADER}Demand states:{bcolors.ENDC}")
        print(tabulate(seller_states, seller_features))

    def show_vehicle_states(self):
        vehicles = self.env.snapshot_list["transport"]

        vehicle_number = len(vehicles)

        vehicle_features = ("id", "facility_id", "location", "steps", "patient", "source", "destination", "payload", "product_id")

        vehicle_states = vehicles[self.env.frame_index::vehicle_features].flatten().reshape(vehicle_number, -1).astype(np.int)

        print(f"{bcolors.HEADER}Vehicle states:{bcolors.ENDC}")

        print(tabulate(vehicle_states, vehicle_features))

    def show_manufacture_states(self):
        # This function is used to debug manufacturing logic.
        manufactures = self.env.snapshot_list["manufacture"]
        storages = self.env.snapshot_list["storage"]

        manufacture_unit_number = len(manufactures)
        storage_unit_number = len(storages)

        # manufacture number for current tick
        features = ("id", "facility_id", "storage_id", "product_id", "product_unit_cost", "production_rate", "manufacturing_number")
        states = manufactures[self.env.frame_index::features].flatten().reshape(manufacture_unit_number, -1).astype(np.int)

        # show bom
        bom_info = []
        for state in states:
            output_product_id = state[3]

            # NOTE: we are using internal data to simplify the code
            sku = self.env._business_engine.world.get_sku_by_id(output_product_id)

            # this sku need source material
            if len(sku.bom) > 0:
                bom_info.append([output_product_id, [s for s in sku.bom.keys()], [s for s in sku.bom.values()]])
            else:
                bom_info.append([output_product_id, "None", "None"])

        print(f"{bcolors.HEADER}SKU bom info:{bcolors.ENDC}")
        print(tabulate(bom_info, ("product id", "source materials", "source material cost per lot")))

        # show manufacture unit data
        print(f"{bcolors.HEADER}Manufacture states:{bcolors.ENDC}")
        print(tabulate(states, headers=features))

        # show storage state to see if product changed
        print(f"{bcolors.HEADER}Manufacture storage states:{bcolors.ENDC}")

        storage_features = ["id", "remaining_space", "capacity"]
        storage_states_summary = []

        for state in states:
            # DO NOTE: this is id, CANNOT be used to query
            facility_id = state[1]
            storage_id = state[2]

            storage_node_name, storage_index = self.env.summary["node_mapping"]["mapping"][storage_id]

            # NOTE: we cannot mix list and normal attribute to query state,
            # we have to query one by one
            product_list = storages[self.env.frame_index:storage_index:"product_list"].flatten().astype(np.int)
            product_number = storages[self.env.frame_index:storage_index:"product_number"].flatten().astype(np.int)
            storage_states = storages[self.env.frame_index:storage_index:storage_features].flatten().astype(np.int)

            cur_storage_states = []
            cur_storage_states.extend(storage_states)
            cur_storage_states.append(product_list)
            cur_storage_states.append(product_number)

            storage_states_summary.append(cur_storage_states)

        print(tabulate(storage_states_summary, headers=storage_features+["product_list", "product_number"]))

    def choose_action(self):
        action = {}

        consumer_actions = self.choose_consumer_action()

        for consumer_action in consumer_actions:
            action[consumer_action.id] = consumer_action

        print(consumer_actions)

        return action

    def choose_consumer_action(self):
        # dummy actions

        # push consumers to generate order

        actions = []

        # check if lack of any source material
        storages = self.env.snapshot_list["storage"]

        for storage_index in range(len(storages)):
            product_list = storages[self.env.frame_index:storage_index:"product_list"].flatten().astype(np.int)
            product_number = storages[self.env.frame_index:storage_index:"product_number"].flatten().astype(np.int)

            for product_id, product_number in zip(product_list, product_number):
                if product_number <= 0:
                    facility_id = storages[self.env.frame_index:storage_index:"facility_id"].flatten()[0]

                    for consumer in self.env.summary["node_mapping"]["detail"][facility_id]["units"]["consumers"]:
                        if consumer["sku_id"] == product_id:
                            sources = self.env.snapshot_list["consumer"][self.env.frame_index:consumer["node_index"]:"sources"]

                            if sources is not None:
                                upstreams = sources.flatten().astype(np.int)

                                actions.append(ConsumerAction(consumer["id"], upstreams[0], 30, 1))

        return actions


def main():
    start_tick = 0
    durations = 100
    env = Env(scenario="supply_chain", topology="sample1", start_tick=start_tick, durations=durations)

    irenv = InteractiveRenderaleEnv(env)

    for ep in range(1):
        irenv.start()

        env.reset()


if __name__ == "__main__":
    main()
