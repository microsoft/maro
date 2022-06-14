# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import io
import os
import pprint
import shutil
import timeit
from typing import List, Set

import yaml
from ilp_agent import IlpAgent

from maro.simulator import Env
from maro.simulator.scenarios.vm_scheduling import DecisionPayload
from maro.simulator.scenarios.vm_scheduling.common import Action
from maro.utils import LogFormat, Logger, convert_dottable

os.environ["LOG_LEVEL"] = "CRITICAL"
FILE_PATH = os.path.split(os.path.realpath(__file__))[0]
CONFIG_PATH = os.path.join(FILE_PATH, "config.yml")
with io.open(CONFIG_PATH, "r") as in_file:
    raw_config = yaml.safe_load(in_file)
    config = convert_dottable(raw_config)

LOG_PATH = os.path.join(FILE_PATH, "log", config.experiment_name)
simulation_logger = Logger(tag="simulation", format_=LogFormat.none, dump_folder=LOG_PATH, dump_mode="w")
ilp_logger = Logger(tag="ilp", format_=LogFormat.none, dump_folder=LOG_PATH, dump_mode="w")

if __name__ == "__main__":
    start_time = timeit.default_timer()
    env = Env(
        scenario=config.env.scenario,
        topology=config.env.topology,
        start_tick=config.env.start_tick,
        durations=config.env.durations,
        snapshot_resolution=config.env.resolution,
    )
    shutil.copy(
        os.path.join(env._business_engine._config_path, "config.yml"),
        os.path.join(LOG_PATH, "BEconfig.yml"),
    )
    shutil.copy(CONFIG_PATH, os.path.join(LOG_PATH, "config.yml"))

    if config.env.seed is not None:
        env.set_seed(config.env.seed)

    metrics: object = None
    decision_event: DecisionPayload = None
    is_done: bool = False
    action: Action = None

    metrics, decision_event, is_done = env.step(None)

    # Get the core & memory capacity of all PMs in this environment.
    pm_capacity = env.snapshot_list["pms"][env.frame_index :: ["cpu_cores_capacity", "memory_capacity"]].reshape(-1, 2)
    pm_num = pm_capacity.shape[0]

    # ILP agent.
    ilp_agent = IlpAgent(
        ilp_config=config.ilp,
        pm_capacity=pm_capacity,
        vm_table_path=env.configs.VM_TABLE,
        env_start_tick=config.env.start_tick,
        env_duration=config.env.durations,
        simulation_logger=simulation_logger,
        ilp_logger=ilp_logger,
        log_path=LOG_PATH,
    )

    while not is_done:
        # Get live VMs in each PM.
        live_vm_set_list: List[Set[int]] = [env._business_engine._machines[idx]._live_vms for idx in range(pm_num)]

        action = ilp_agent.choose_action(
            env_tick=env.tick,
            cur_vm_id=decision_event.vm_id,
            live_vm_set_list=live_vm_set_list,
        )

        metrics, decision_event, is_done = env.step(action)

    end_time = timeit.default_timer()
    simulation_logger.info(
        f"[Offline ILP] Topology: {config.env.topology}. Total ticks: {config.env.durations}."
        f" Start tick: {config.env.start_tick}.",
    )
    simulation_logger.info(f"[Timer] {end_time - start_time:.2f} seconds to finish the simulation.")
    ilp_agent.report_allocation_summary()
    simulation_logger.info(pprint.pformat(metrics._original_dict))
