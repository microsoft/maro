from typing import Dict, Optional, Tuple

import numpy as np

from examples.rl.cim.config import action_shaping_conf
from maro.rl_v3.learning import AbsEnvSampler, ActionWithAux, CacheElement, SimpleAgentWrapper
from maro.simulator import Env
from maro.simulator.scenarios.cim.common import Action, ActionType
from .config import env_conf, port_attributes, reward_shaping_conf, state_shaping_conf, vessel_attributes
from .policies import get_policy_func_dict
from .trainers import get_trainer_func_dict, policy2trainer


class CIMEnvSampler(AbsEnvSampler):
    def _get_global_and_agent_state(
        self, event, tick: int = None
    ) -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray]]:
        tick = self._env.tick
        vessel_snapshots, port_snapshots = self._env.snapshot_list["vessels"], self._env.snapshot_list["ports"]
        port_idx, vessel_idx = event.port_idx, event.vessel_idx
        ticks = [max(0, tick - rt) for rt in range(state_shaping_conf["look_back"] - 1)]
        future_port_list = vessel_snapshots[tick: vessel_idx: 'future_stop_list'].astype('int')
        state = np.concatenate([
            port_snapshots[ticks: [port_idx] + list(future_port_list): port_attributes],
            vessel_snapshots[tick: vessel_idx: vessel_attributes]
        ])
        return np.zeros(0), {port_idx: state}

    def _translate_to_env_action(self, action_with_aux_dict: Dict[str, ActionWithAux], event) -> Dict[str, object]:
        action_space = action_shaping_conf["action_space"]
        finite_vsl_space = action_shaping_conf["finite_vessel_space"]
        has_early_discharge = action_shaping_conf["has_early_discharge"]

        port_idx, action_with_aux = list(action_with_aux_dict.items()).pop()
        assert isinstance(port_idx, str)
        assert isinstance(action_with_aux, ActionWithAux)

        vsl_idx, action_scope = event.vessel_idx, event.action_scope
        vsl_snapshots = self._env.snapshot_list["vessels"]
        vsl_space = vsl_snapshots[self._env.tick:vsl_idx:vessel_attributes][2] if finite_vsl_space else float("inf")

        model_action = action_with_aux.action
        percent = abs(action_space[model_action])
        zero_action_idx = len(action_space) / 2  # index corresponding to value zero.
        if model_action < zero_action_idx:
            action_type = ActionType.LOAD
            actual_action = min(round(percent * action_scope.load), vsl_space)
        elif model_action > zero_action_idx:
            action_type = ActionType.DISCHARGE
            early_discharge = vsl_snapshots[self._env.tick:vsl_idx:"early_discharge"][0] if has_early_discharge else 0
            plan_action = percent * (action_scope.discharge + early_discharge) - early_discharge
            actual_action = round(plan_action) if plan_action > 0 else round(percent * action_scope.discharge)
        else:
            actual_action, action_type = 0, None

        return {port_idx: Action(vsl_idx, int(port_idx), actual_action, action_type)}

    def _get_reward(self, env_action_dict: Dict[str, object], tick: int) -> Dict[str, float]:
        start_tick = tick + 1
        ticks = list(range(start_tick, start_tick + reward_shaping_conf["time_window"]))

        # Get the ports that took actions at the given tick
        ports = [int(port) for port in list(env_action_dict.keys())]
        port_snapshots = self._env.snapshot_list["ports"]
        future_fulfillment = port_snapshots[ticks:ports:"fulfillment"].reshape(len(ticks), -1)
        future_shortage = port_snapshots[ticks:ports:"shortage"].reshape(len(ticks), -1)

        decay_list = [reward_shaping_conf["time_decay"] ** i for i in range(reward_shaping_conf["time_window"])]
        rewards = np.float32(
            reward_shaping_conf["fulfillment_factor"] * np.dot(future_fulfillment.T, decay_list)
            - reward_shaping_conf["shortage_factor"] * np.dot(future_shortage.T, decay_list)
        )
        return {agent_id: reward for agent_id, reward in zip(ports, rewards)}

    def _post_step(self, cache_element: CacheElement, reward: Dict[str, float]) -> None:
        self._tracker["env_metric"] = self._env.metrics


def get_env_sampler():
    algorithm = "dqn"
    return CIMEnvSampler(
        get_env_func=lambda: Env(**env_conf),
        get_policy_func_dict=get_policy_func_dict,
        get_trainer_func_dict=get_trainer_func_dict,
        agent2policy={agent: f"{algorithm}.{agent}" for agent in Env(**env_conf).agent_idx_list},
        policy2trainer=policy2trainer,
        agent_wrapper_cls=SimpleAgentWrapper,
        return_experiences=False
    )


if __name__ == "__main__":
    env_sampler = get_env_sampler()
