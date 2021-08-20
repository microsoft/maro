# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

from maro.rl.policy import ACActionInfo
from maro.rl.wrappers import AbsEnvWrapper
from maro.simulator import Env
from maro.simulator.scenarios.cim.common import Action, ActionType


def post_step(env, tracker, transition):
    tracker["env_metric"] = env.metrics


class CIMEnvWrapper(AbsEnvWrapper):
    def __init__(
        self, env, replay_agent_ids=None, *, port_attributes, vessel_attributes, num_actions, look_back,
        max_ports_downstream, reward_eval_delay, fulfillment_factor, shortage_factor, time_decay,
        finite_vessel_space=True, has_early_discharge=True
    ):
        super().__init__(
            env,
            replay_agent_ids=replay_agent_ids,
            reward_eval_delay=reward_eval_delay,
            post_step=post_step
        )
        self.port_attributes = port_attributes
        self.vessel_attributes = vessel_attributes
        self.action_space = list(np.linspace(-1.0, 1.0, num_actions))
        self.look_back = look_back
        self.max_ports_downstream = max_ports_downstream
        self.fulfillment_factor = fulfillment_factor
        self.shortage_factor = shortage_factor
        self.time_decay = time_decay
        self.finite_vessel_space = finite_vessel_space
        self.has_early_discharge = has_early_discharge
        self._last_action_tick = None
        self._state_dim = (
            (self.look_back + 1) * (self.max_ports_downstream + 1) * len(self.port_attributes)
            + len(self.vessel_attributes)
        )
        self._state_info = None

    @property
    def state_dim(self):
        return self._state_dim

    def get_state(self, tick=None):
        if tick is None:
            tick = self.env.tick
        vessel_snapshots, port_snapshots = self.env.snapshot_list["vessels"], self.env.snapshot_list["ports"]
        port_idx, vessel_idx = self.event.port_idx, self.event.vessel_idx
        ticks = [max(0, tick - rt) for rt in range(self.look_back - 1)]
        future_port_idx_list = vessel_snapshots[tick: vessel_idx: 'future_stop_list'].astype('int')
        port_features = port_snapshots[ticks: [port_idx] + list(future_port_idx_list): self.port_attributes]
        vessel_features = vessel_snapshots[tick: vessel_idx: self.vessel_attributes]
        self._state_info = {
            port_idx: {"tick": tick, "action_scope": self.event.action_scope, "vessel_idx": vessel_idx}
        }
        state = np.concatenate((port_features, vessel_features))
        self._last_action_tick = tick
        return {port_idx: state}

    def to_env_action(self, action_by_agent: dict):
        env_action = []
        for agent_id, action_info in action_by_agent.items():
            tick = self._state_info[agent_id]["tick"]
            vessel = self._state_info[agent_id]["vessel_idx"]
            action_scope = self._state_info[agent_id]["action_scope"]
            vessel_snapshots = self.env.snapshot_list["vessels"]
            vessel_space = (
                vessel_snapshots[tick:vessel:self.vessel_attributes][2] if self.finite_vessel_space else float("inf")
            )
            early_discharge = vessel_snapshots[tick:vessel:"early_discharge"][0] if self.has_early_discharge else 0

            model_action = action_info.action if isinstance(action_info, ACActionInfo) else action_info
            percent = abs(self.action_space[model_action])
            zero_action_idx = len(self.action_space) / 2  # index corresponding to value zero.
            if model_action < zero_action_idx:
                action_type = ActionType.LOAD
                actual_action = min(round(percent * action_scope.load), vessel_space)
            elif model_action > zero_action_idx:
                action_type = ActionType.DISCHARGE
                plan_action = percent * (action_scope.discharge + early_discharge) - early_discharge
                actual_action = round(plan_action) if plan_action > 0 else round(percent * action_scope.discharge)
            else:
                actual_action, action_type = 0, None

            env_action.append(
                Action(port_idx=agent_id, vessel_idx=vessel, quantity=actual_action, action_type=action_type)
            )

        return env_action

    def get_reward(self, actions, tick=None):
        """Delayed reward evaluation."""
        if tick is None:
            tick = self._last_action_tick
        start_tick = tick + 1
        ticks = list(range(start_tick, start_tick + self.reward_eval_delay))

        # Get the ports that took actions at the given tick
        ports = [action.port_idx for action in actions]
        port_snapshots = self.env.snapshot_list["ports"]
        future_fulfillment = port_snapshots[ticks:ports:"fulfillment"].reshape(len(ticks), -1)
        future_shortage = port_snapshots[ticks:ports:"shortage"].reshape(len(ticks), -1)

        decay_list = [self.time_decay ** i for i in range(self.reward_eval_delay)]
        rewards = np.float32(
            self.fulfillment_factor * np.dot(future_fulfillment.T, decay_list)
            - self.shortage_factor * np.dot(future_shortage.T, decay_list)
        )
        return {agent_id: reward for agent_id, reward in zip(ports, rewards)}


env_config = {
    "basic": {
        "scenario": "cim",
        "topology": "toy.4p_ssdd_l0.0",
        "durations": 560
    },
    "wrapper": {
        "port_attributes": ["empty", "full", "on_shipper", "on_consignee", "booking", "shortage", "fulfillment"],
        "vessel_attributes": ["empty", "full", "remaining_space"],
        "num_actions": 21,
        # Parameters for computing states
        "look_back": 7,
        "max_ports_downstream": 2,
        # Parameters for computing actions
        "finite_vessel_space": True,
        "has_early_discharge": True,
        # Parameters for computing rewards
        "reward_eval_delay": 99,
        "fulfillment_factor": 1.0,
        "shortage_factor": 1.0,
        "time_decay": 0.97
    }
}

def get_env_wrapper(replay_agent_ids=None):
    return CIMEnvWrapper(Env(**env_config["basic"]), replay_agent_ids=replay_agent_ids, **env_config["wrapper"]) 
 
# obtain state dimension from a temporary env_wrapper instance
STATE_DIM = get_env_wrapper().state_dim