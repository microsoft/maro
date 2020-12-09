import io
import os
import random
import timeit

import yaml

from maro.simulator import Env
from maro.simulator.scenarios.data_center import AssignAction, DecisionPayload, PostponeAction
from maro.utils import convert_dottable

CONFIG_PATH = os.path.join(os.path.split(os.path.realpath(__file__))[0], "config.yml")
with io.open(CONFIG_PATH, "r") as in_file:
    raw_config = yaml.safe_load(in_file)
    config = convert_dottable(raw_config)


if __name__ == "__main__":
    start_time = timeit.default_timer()

    env = Env(
        scenario=config.env.scenario,
        topology=config.env.topology,
        start_tick=config.env.start_tick,
        durations=config.env.durations,
        snapshot_resolution=config.env.resolution
    )

    if config.env.seed is not None:
        env.set_seed(config.env.seed)
        random.seed(config.env.seed)

    metrics: object = None
    decision_event: DecisionPayload = None
    is_done: bool = False
    action: AssignAction = None
    metrics, decision_event, is_done = env.step(None)
    while not is_done:
        valid_pm_num: int = len(decision_event.valid_pms)
        if valid_pm_num <= 0:
            # No valid PM now, postpone.
            action: PostponeAction = PostponeAction(
                vm_id=decision_event.vm_id,
                # why remaining_buffer_time in this payload?
                remaining_buffer_time=decision_event.remaining_buffer_time - 1
            )
        else:
            # Randomly choose a vailable PM.
            random_idx = random.randint(0, valid_pm_num - 1)
            pm_id = decision_event.valid_pms[random_idx].pm_id
            action: AssignAction = AssignAction(
                vm_id=decision_event.vm_id,
                # why remaining_buffer_time in this payload?
                remaining_buffer_time=decision_event.remaining_buffer_time,
                pm_id=pm_id
            )
        metrics, decision_event, is_done = env.step(action)

    end_time = timeit.default_timer()
    print(f"[Timer] {end_time - start_time:.2f} seconds is used for ...")
    print(metrics)
