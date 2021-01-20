import io
import os
import pdb
import random
import timeit

import yaml
import importlib

from maro.simulator import Env
from maro.simulator.scenarios.vm_scheduling import AllocateAction, DecisionPayload, PostponeAction
from maro.utils import convert_dottable

def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod

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

    agent_class = import_class(config.alg.type)
    if config.alg.args is None:
        agent = agent_class()
    else:
        agent = agent_class(**config.alg.args)

    if config.env.seed is not None:
        env.set_seed(config.env.seed)
        random.seed(config.env.seed)

    metrics, decision_event, is_done = env.step(None)

    while not is_done:
        action = agent.choose_action(decision_event, env)
        metrics, decision_event, is_done = env.step(action)

    end_time = timeit.default_timer()
    print(
        f"[Best fit] Topology: {config.env.topology}. Total ticks: {config.env.durations}."
        f" Start tick: {config.env.start_tick}."
    )
    print(f"[Timer] {end_time - start_time:.2f} seconds to finish the simulation.")
    print(metrics)
