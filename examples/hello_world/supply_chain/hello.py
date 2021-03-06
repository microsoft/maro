# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import tcod
import pprint

from tabulate import tabulate
from maro.simulator import Env
from maro.simulator.scenarios.supply_chain import ConsumerAction


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

    def start(self):
        print(f"{bcolors.WARNING}Press SPACE for next step!!!{bcolors.ENDC}")
        print(f"{bcolors.WARNING}Press S to show current debug states!!!{bcolors.ENDC}")
        print(f"{bcolors.WARNING}Press M to show env summary!!!{bcolors.ENDC}")

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
                        if event.sym == tcod.event.K_SPACE :
                            # push environment to next step
                            metrics, decision_event, is_done = self.env.step(None)

                            print(f"{bcolors.OKGREEN}Current environment tick:", self.env.tick, f"{bcolors.ENDC}")

                            if is_done:
                                return
                        elif event.sym == tcod.event.K_s:
                            # show state we predefined
                            self.show_states()
                        elif event.sym == tcod.event.K_m:
                            self.show_summary()

    def show_summary(self):
        pp = pprint.PrettyPrinter(indent=2, depth=8)

        print(f"{bcolors.HEADER}env summary:{bcolors.ENDC}")

        pp.pprint(self.env.summary)

    def show_states(self):
        # print("total snapshots:\n", len(self.env.snapshot_list))
        # print("transport patient:\n", self.env.snapshot_list["transport"][:0:"patient"].flatten())

        # since the seller node number will not change, we can reshape it as below
        # seller_number = len(self.env.snapshot_list["seller"])
        # print("seller demand:\n", self.env.snapshot_list["seller"][::"demand"].flatten().reshape((-1, seller_number)))

        self.show_manufacture_states()

    def show_manufacture_states(self):
        # This function is used to debug manufacturing logic.
        manufactures = self.env.snapshot_list["manufacture"]
        storages = self.env.snapshot_list["storage"]

        manufacture_unit_number = len(manufactures)
        storage_unit_number = len(storages)

        # manufacture number for current tick
        features = ("id", "facility_id", "storage_id", "output_product_id", "product_unit_cost", "production_rate", "manufacturing_number")
        states = manufactures[self.env.frame_index::features].flatten().reshape(manufacture_unit_number, -1).astype(np.int)

        # show manufacture unit data
        print(f"{bcolors.HEADER}Manufacture states:{bcolors.ENDC}")
        print(tabulate(states, headers=features))

        # show storage state to see if product changed
        print(f"{bcolors.HEADER}Storage states:{bcolors.ENDC}")

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

            # print(product_list)
            # print(product_number)
            # print(storage_states)

            cur_storage_states = []
            cur_storage_states.extend(storage_states)
            cur_storage_states.append(product_list)
            cur_storage_states.append(product_number)

            storage_states_summary.append(cur_storage_states)

        print(tabulate(storage_states_summary, headers=storage_features+["product_list", "product_number"]))


    def choose_action(self):
        # dummy actions

        # push consumers to generate order
        # consumers =
        pass


def main():
    start_tick = 0
    durations = 100
    env = Env(scenario="supply_chain", topology="no", start_tick=start_tick, durations=durations)

    irenv = InteractiveRenderaleEnv(env)

    for ep in range(1):
        irenv.start()

        env.reset()


if __name__ == "__main__":
    main()
