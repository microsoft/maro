from typing import Any, Dict, Optional, Tuple

import numpy as np

import torch

from examples.cim.rl_v31.config import action_shaping_conf, env_conf, port_attributes, reward_shaping_conf, \
    state_shaping_conf, \
    vessel_attributes
from maro.rl_v31.rollout.wrapper import EnvWrapper
from maro.simulator import Env
from maro.simulator.scenarios.cim.common import Action, ActionType, DecisionEvent

env = Env(**env_conf)


class CimEnvWrapper(EnvWrapper):
    def __init__(
        self,
        env: Env,
        reward_eval_delay: Optional[int] = None,
        max_episode_length: Optional[int] = None,  # TODO: move to collector?
    ) -> None:
        super().__init__(
            env=env,
            reward_eval_delay=reward_eval_delay,
            max_episode_length=max_episode_length,
            discard_tail_elements=True,
        )

    def state_to_obs(self, event: DecisionEvent, tick: int = None) -> Tuple[np.ndarray, Dict[Any, np.ndarray]]:
        tick = self.env.tick
        vessel_snapshots, port_snapshots = self.env.snapshot_list["vessels"], self.env.snapshot_list["ports"]
        port_idx, vessel_idx = event.port_idx, event.vessel_idx
        ticks = [max(0, tick - rt) for rt in range(state_shaping_conf["look_back"] - 1)]
        future_port_list = vessel_snapshots[tick:vessel_idx:"future_stop_list"].astype("int")
        state = np.concatenate(
            [
                port_snapshots[ticks : [port_idx] + list(future_port_list) : port_attributes],
                vessel_snapshots[tick:vessel_idx:vessel_attributes],
            ],
        )
        return state, {port_idx: state}

    def policy_act_to_act(
        self,
        event: DecisionEvent,
        policy_act_dict: Dict[Any, torch.Tensor],
        tick: int = None,
    ) -> Dict[Any, Action]:
        action_space = action_shaping_conf["action_space"]
        finite_vsl_space = action_shaping_conf["finite_vessel_space"]
        has_early_discharge = action_shaping_conf["has_early_discharge"]

        port_idx, model_action = list(policy_act_dict.items()).pop()

        vsl_idx, action_scope = event.vessel_idx, event.action_scope
        vsl_snapshots = self.env.snapshot_list["vessels"]
        vsl_space = vsl_snapshots[self.env.tick : vsl_idx : vessel_attributes][2] if finite_vsl_space else float("inf")

        percent = abs(action_space[model_action.item()])
        zero_action_idx = len(action_space) / 2  # index corresponding to value zero.
        if model_action < zero_action_idx:
            action_type = ActionType.LOAD
            actual_action = min(round(percent * action_scope.load), vsl_space)
        elif model_action > zero_action_idx:
            action_type = ActionType.DISCHARGE
            early_discharge = (
                vsl_snapshots[self.env.tick : vsl_idx : "early_discharge"][0] if has_early_discharge else 0
            )
            plan_action = percent * (action_scope.discharge + early_discharge) - early_discharge
            actual_action = round(plan_action) if plan_action > 0 else round(percent * action_scope.discharge)
        else:
            actual_action, action_type = 0, None

        return {port_idx: Action(vsl_idx, int(port_idx), actual_action, action_type)}

    def get_reward(self, event: DecisionEvent, act_dict: Dict[Any, ActionType], tick: int) -> Dict[Any, float]:
        start_tick = tick + 1
        ticks = list(range(start_tick, start_tick + reward_shaping_conf["time_window"]))

        # Get the ports that took actions at the given tick
        ports = [int(port) for port in list(act_dict.keys())]
        port_snapshots = self.env.snapshot_list["ports"]
        future_fulfillment = port_snapshots[ticks:ports:"fulfillment"].reshape(len(ticks), -1)
        future_shortage = port_snapshots[ticks:ports:"shortage"].reshape(len(ticks), -1)

        decay_list = [reward_shaping_conf["time_decay"] ** i for i in range(reward_shaping_conf["time_window"])]
        rewards = np.float32(
            reward_shaping_conf["fulfillment_factor"] * np.dot(future_fulfillment.T, decay_list)
            - reward_shaping_conf["shortage_factor"] * np.dot(future_shortage.T, decay_list),
        )
        return {agent_id: reward for agent_id, reward in zip(ports, rewards)}

    def gather_info(self) -> dict:
        return self.env.metrics
