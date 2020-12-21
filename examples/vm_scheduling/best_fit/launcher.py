import io
import os
import random
import timeit

import yaml

from maro.simulator import Env
from maro.simulator.scenarios.vm_scheduling import DecisionPayload, PlaceAction, PostponeAction
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
    action: PlaceAction = None
    metrics, decision_event, is_done = env.step(None)

    while not is_done:
        valid_pm_num: int = len(decision_event.valid_pms)
        if valid_pm_num <= 0:
            # No valid PM now, postpone.
            action: PostponeAction = PostponeAction(
                vm_id=decision_event.vm_id,
                postpone_frequency=1
            )
        else:
            # Get the capacity and allocated cores from snapshot.
            valid_pm_info = env.snapshot_list["pms"][
                env.frame_index:decision_event.valid_pms:["cpu_cores_capacity", "cpu_cores_allocated"]
            ].reshape(-1, 2)
            # Calculate to get the remaining cpu cores.
            cpu_cores_remaining = valid_pm_info[:, 0] - valid_pm_info[:, 1]
            # Choose the one with the closet remaining CPU.
            chosen_idx = 0
            minimum_remaining_cpu_cores = cpu_cores_remaining[0]
            for i, remaining in enumerate(cpu_cores_remaining):
                if remaining < minimum_remaining_cpu_cores:
                    chosen_idx = i
                    minimum_remaining_cpu_cores = remaining
            # Take action to place on the closet pm.
            action: PlaceAction = PlaceAction(
                vm_id=decision_event.vm_id,
                pm_id=decision_event.valid_pms[chosen_idx]
            )
        metrics, decision_event, is_done = env.step(action)

    end_time = timeit.default_timer()
    print(
        f"[Best fit] Topology: {config.env.topology}. Total ticks: {config.env.durations}."
        f" Start tick: {config.env.start_tick}."
    )
    print(f"[Timer] {end_time - start_time:.2f} seconds to finish the simulation.")
    print(metrics)
