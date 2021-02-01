# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.simulator.scenarios.cim.common import Action, DecisionEvent
from maro.vector_env import VectorEnv


if __name__ == "__main__":
    with VectorEnv(batch_num=4, scenario="cim", topology="toy.5p_ssddd_l0.0", durations=100) as env:
        for ep in range(2):
            print("current episode:", ep)

            metrics, decision_event, is_done = (None, None, False)

            while not is_done:
                action = None

                # Usage:
                # 1. Only push speicified (1st for this example) environment, leave others behind
                # if decision_event:
                #     env0_dec: DecisionEvent = decision_event[0]

                #     # 1.1 After 1st environment is done, then others will push forward.
                #     if env0_dec:
                #         ss0 = env.snapshot_list["vessels"][env0_dec.tick:env0_dec.vessel_idx:"remaining_space"]
                #         action = {0: Action(env0_dec.vessel_idx, env0_dec.port_idx, -env0_dec.action_scope.load)}

                # 2. Only pass action to 1st environment (give None to other environments),
                # but keep pushing all the environment, until the end
                if decision_event:
                    env0_dec: DecisionEvent = decision_event[0]

                    if env0_dec:
                        ss0 = env.snapshot_list["vessels"][env0_dec.tick:env0_dec.vessel_idx:"remaining_space"]

                        action = [None] * env.batch_number

                        # with a list of action, will push all environment to next step
                        action[0] = Action(env0_dec.vessel_idx, env0_dec.port_idx, -env0_dec.action_scope.load)

                metrics, decision_event, is_done = env.step(action)

            print("Final tick for each environment:", env.tick)
            print("Final frame index for each environment:", env.frame_index)

            env.reset()
