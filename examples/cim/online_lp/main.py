from collections import OrderedDict
import io
import os
import random

import numpy as np
import shutil
import yaml

from maro.simulator import Env
from maro.utils import DottableDict, Logger, LogFormat, convert_dottable

from agent import LPAgent
from forecaster import Forecaster
from online_lp import OnlineLP


FILE_PATH = os.path.split(os.path.realpath(__file__))[0]
CONFIG_PATH = os.path.join(FILE_PATH, "config.yml")
with io.open(CONFIG_PATH, "r") as in_file:
    raw_config = yaml.safe_load(in_file)
    config = convert_dottable(raw_config)

LOG_FOLDER = os.path.join(FILE_PATH, "log", config.experiment_name)
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)
shutil.copy(CONFIG_PATH, LOG_FOLDER)


class Runner:
    def __init__(
        self,
        env_config: DottableDict,
        ilp_config: DottableDict
    ):
        self._random_seed = env_config.seed
        self._set_seed()

        self._durations = env_config.durations

        self._env = Env(
            scenario=env_config.scenario,
            topology=env_config.topology,
            start_tick=env_config.start_tick,
            durations=env_config.durations
        )
        self._port_idx2name = {idx: name for name, idx in self._env.summary["node_mapping"]["ports"].items()}
        self._vessel_idx2name = {idx: name for name, idx in self._env.summary["node_mapping"]["vessels"].items()}

        self._online_lp_agent = self._load_agent(ilp_config)

        self._init_loggers()

    def _set_seed(self):
        np.random.seed(self._random_seed)
        random.seed(self._random_seed)

    def _init_loggers(self):
        self._logger = Logger(
            tag="runner",
            format_=LogFormat.simple,
            dump_folder=LOG_FOLDER,
            dump_mode="w",
            auto_timestamp=False
        )

        self._port_logger = dict()
        for port in self._port_idx2name.keys():
            self._port_logger[port] = Logger(
                tag=f"{self._port_idx2name[port]}.logger",
                format_=LogFormat.none,
                dump_folder=LOG_FOLDER,
                dump_mode="w",
                extension_name="csv",
                auto_timestamp=False
            )
            self._port_logger[port].debug("tick, action, port_inventory, shortage")

    def _load_agent(self, config: DottableDict):
        online_lp = OnlineLP(
            port_idx2name=self._port_idx2name,
            vessel_idx2name=self._vessel_idx2name,
            topology_config=self._env.configs,
            lp_config=config
        )

        forecaster = Forecaster(
            moving_average_length=config.moving_average_length,
            port_idx2name=self._port_idx2name,
            vessel_idx2name=self._vessel_idx2name,
            topology_config=self._env.configs,
        )

        agent = LPAgent(
            algorithm=online_lp,
            forecaster=forecaster,
            vessel_idx2name=self._vessel_idx2name,
            window_size=config.window_size
        )
        return agent

    def start(self):
        self._set_seed()

        _, decision_event, is_done = self._env.step(None)

        while not is_done:
            initial_port_empty, initial_vessel_empty, initial_vessel_full = self._get_initial_values()

            action = self._online_lp_agent.choose_action(
                decision_event=decision_event,
                finished_events=self._env._event_buffer.get_finished_events(),
                snapshot_list=self._env.snapshot_list,
                initial_port_empty=initial_port_empty,
                initial_vessel_empty=initial_vessel_empty,
                initial_vessel_full=initial_vessel_full
            )

            ports = self._env.snapshot_list["ports"]
            self._port_logger[decision_event.port_idx].info(
                f"{decision_event.tick}, "
                f"{action.quantity}, "
                f"{ports[decision_event.tick:decision_event.port_idx:'empty'][0]}, "
                f"{np.sum(ports[:decision_event.port_idx:'shortage'])}"
            )

            _, decision_event, is_done = self._env.step(action)

        self._print_summary()

        self._env.reset()

        self._online_lp_agent.reset()

    def _get_initial_values(self):
        initial_port_empty = {
            port.name: port.empty
            for port in self._env._business_engine._ports
        }

        initial_vessel_empty = {
            vessel.name: vessel.empty
            for vessel in self._env._business_engine._vessels
        }

        initial_vessel_full = {
            vessel.name: {
                port.name: self._env._business_engine._full_on_vessels[vessel.idx: port.idx]
                for port in self._env._business_engine._ports
            }
            for vessel in self._env._business_engine._vessels
        }

        return initial_port_empty, initial_vessel_empty, initial_vessel_full

    def _get_pretty_info(self, port_attribute):
        data_list = self._env.snapshot_list["ports"][
            self._env.tick:self._env.agent_idx_list:port_attribute
        ]
        pretty_dict = OrderedDict({self._port_idx2name[i]: data for i, data in enumerate(data_list)})
        total = np.sum(data_list)
        return pretty_dict, total

    def _print_summary(self):
        pretty_shortage_dict, total_shortage = self._get_pretty_info("acc_shortage")

        pretty_booking_dict, total_booking = self._get_pretty_info("acc_booking")

        self._logger.critical(
            f"{self._env.name} | {config.experiment_name} | total ticks: {self._durations}, "
            f"total booking: {total_booking}, total shortage: {total_shortage}"
        )

        last_224_shortage = total_shortage - np.sum(
            self._env.snapshot_list["ports"][self._env.tick - 224:self._env.agent_idx_list:"acc_shortage"]
        )
        print(f"last 224 shortage: {last_224_shortage}")


if __name__ == "__main__":
    runner = Runner(env_config=config.env, ilp_config=config.online_lp)
    runner.start()
