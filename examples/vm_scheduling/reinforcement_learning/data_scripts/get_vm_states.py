import io
import os
import random
import timeit

import yaml
import importlib

from maro.simulator import Env
from maro.utils import convert_dottable


CONFIG_PATH = os.path.join(os.path.split(os.path.realpath(__file__))[0], "config.yml")
with io.open(CONFIG_PATH, "r") as in_file:
    raw_config = yaml.safe_load(in_file)
    config = convert_dottable(raw_config)

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

    vm_states = []
    metrics, decision_event, is_done = env.step(None)

    while not is_done:
        action = PostponeAction(
            vm_id=decision_event.vm_id,
            postpone_step=1
        )
        metrics, decision_event, is_done = env.step(action)

        vm_info = [
            decision_event.vm_cpu_cores_requirement,
            decision_event.vm_memory_requirement,
            decision_event.vm_lifetime,
            env.tick,
            decision_event.vm_unit_price
        ]

        vm_states.append(vm_info)

    np.save("vm_states.npy", np.array(vm_states))
