from copy import copy

import numpy as np
import torch

from maro.rl import AbsAgentManager, AgentMode
from maro.utils import DummyLogger

from .actor_critic import ActorCritic
from .agent import TrainableAgent
from .numpy_store import NumpyStore
from .simple_gnn import SharedAC
from .state_shaper import GNNStateShaper


class SimpleAgentManger(AbsAgentManager):
    def __init__(
            self, name, agent_id_list, port_code_list, vessel_code_list, demo_env, state_shaper: GNNStateShaper,
            logger=DummyLogger()):
        super().__init__(
            name, AgentMode.TRAIN, agent_id_list, state_shaper=state_shaper, action_shaper=None,
            experience_shaper=None, explorer=None)
        self.port_code_list = copy(port_code_list)
        self.vessel_code_list = copy(vessel_code_list)
        self.demo_env = demo_env
        self._logger = logger

    def assemble(self, config):
        v_dim, vedge_dim = self._state_shaper.get_input_dim("v"), self._state_shaper.get_input_dim("vedge")
        p_dim, pedge_dim = self._state_shaper.get_input_dim("p"), self._state_shaper.get_input_dim("pedge")

        self.device = torch.device(config.training.device)
        self._logger.info(config.training.device)
        ac_model = SharedAC(
            p_dim, pedge_dim, v_dim, vedge_dim, config.model.tick_buffer, config.model.action_dim).to(self.device)

        value_dict = {
            ("s", "v"):
                (
                    (config.model.tick_buffer, len(self.vessel_code_list), self._state_shaper.get_input_dim("v")),
                    np.float32, False),
            ("s", "p"):
                (
                    (config.model.tick_buffer, len(self.port_code_list), self._state_shaper.get_input_dim("p")),
                    np.float32, False),
            ("s", "vo"): ((len(self.vessel_code_list), len(self.port_code_list)), np.int64, True),
            ("s", "po"): ((len(self.port_code_list), len(self.vessel_code_list)), np.int64, True),
            ("s", "vedge"):
                (
                    (len(self.vessel_code_list), len(self.port_code_list), self._state_shaper.get_input_dim("vedge")),
                    np.float32, True),
            ("s", "pedge"):
                (
                    (len(self.port_code_list), len(self.vessel_code_list), self._state_shaper.get_input_dim("vedge")),
                    np.float32, True),
            ("s", "ppedge"):
                (
                    (len(self.port_code_list), len(self.port_code_list), self._state_shaper.get_input_dim("pedge")),
                    np.float32, True),
            ("s", "mask"): ((config.model.tick_buffer, ), np.bool, True),

            ("s_", "v"):
                (
                    (config.model.tick_buffer, len(self.vessel_code_list), self._state_shaper.get_input_dim("v")),
                    np.float32, False),
            ("s_", "p"):
                (
                    (config.model.tick_buffer, len(self.port_code_list), self._state_shaper.get_input_dim("p")),
                    np.float32, False),
            ("s_", "vo"): ((len(self.vessel_code_list), len(self.port_code_list)), np.int64, True),
            ("s_", "po"):
                (
                    (len(self.port_code_list), len(self.vessel_code_list)), np.int64, True),
            ("s_", "vedge"):
                (
                    (len(self.vessel_code_list), len(self.port_code_list), self._state_shaper.get_input_dim("vedge")),
                    np.float32, True),
            ("s_", "pedge"):
                (
                    (len(self.port_code_list), len(self.vessel_code_list), self._state_shaper.get_input_dim("vedge")),
                    np.float32, True),
            ("s_", "ppedge"):
                (
                    (len(self.port_code_list), len(self.port_code_list), self._state_shaper.get_input_dim("pedge")),
                    np.float32, True),
            ("s_", "mask"): ((config.model.tick_buffer, ), np.bool, True),

            # To identify one dimension variable.
            ("R",): ((len(self.port_code_list), ), np.float32, True),
            ("a",): (tuple(), np.int64, True),
        }

        self._algorithm = ActorCritic(
            ac_model, self.device, td_steps=config.training.td_steps, p2p_adj=self._state_shaper.p2p_static_graph,
            gamma=config.training.gamma, learning_rate=config.training.learning_rate)

        for agent_id, cnt in config.env.exp_per_ep.items():
            experience_pool = NumpyStore(value_dict, config.training.parallel_cnt * config.training.train_freq * cnt)
            self._agent_dict[agent_id] = TrainableAgent(agent_id, self._algorithm, experience_pool, self._logger)

    def choose_action(self, agent_id, state):
        return self._agent_dict[agent_id].choose_action(state)

    def load_models_from_files(self, model_pth):
        self._algorithm.load_model(model_pth)

    def train(self, training_config):
        for agent in self._agent_dict.values():
            agent.train(training_config)

    def store_experiences(self, experiences):
        for code, exp_list in experiences.items():
            self._agent_dict[code].store_experiences(exp_list)

    def save_model(self, pth, id):
        self._algorithm.save_model(pth, id)

    def load_model(self, pth):
        self._algorithm.load_model(pth)
