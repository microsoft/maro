# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tcod

from maro.simulator import Env
from maro.simulator.scenarios.cim.common import Action


WINDOW_SCALE_FACTOR = 2
TILESET_SIZE = 8

CHAR_DEFAULT = 0x2610


class InteractiveRenderaleEnv:
    def __init__(self, env: Env):
        self.env = env

    def start(self):
        print("Press SPACE for next step!!!")
        print("Press S for states!!!")

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

                            print("Current environment tick:", self.env.tick)

                            if is_done:
                                return
                        elif event.sym == tcod.event.K_s:
                            # show state we predefined
                            self.show_states()

    def show_states(self):
        print("total snapshots:\n", len(self.env.snapshot_list))
        print("transport patient:\n", self.env.snapshot_list["transport"][:0:"patient"].flatten())

        # since the seller node number will not change, we can reshape it as below
        seller_number = len(self.env.snapshot_list["seller"])
        print("seller demand:\n", self.env.snapshot_list["seller"][::"demand"].flatten().reshape((-1, seller_number)))

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
