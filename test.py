# Import necessary packages
from typing import Any, Dict, List, Tuple, Union, Optional

import numpy as np
import torch
from torch.optim import Adam, RMSprop

from maro.rl.model import DiscreteACBasedNet, FullyConnected, VNet
from maro.rl.policy import DiscretePolicyGradient
from maro.rl.rl_component.rl_component_bundle import RLComponentBundle
from maro.rl.rollout import AbsEnvSampler, CacheElement, ExpElement
from maro.rl.training import TrainingManager
from maro.rl.training.algorithms.trpo import TRPOTrainer, TRPOParams
from maro.rl.training.algorithms.ppo import PPOTrainer, PPOParams
from maro.simulator import Env
from maro.simulator.scenarios.cim.common import Action, ActionType, DecisionEvent
from maro.rl.training.algorithms.ppo import PPOTrainer, PPOParams


class CIMEnvSampler(AbsEnvSampler):
    def _get_global_and_agent_state_impl(
        self, event: DecisionEvent, tick: int = None,
    ) -> Tuple[Union[None, np.ndarray, List[object]], Dict[Any, Union[np.ndarray, List[object]]]]:
        tick = self._env.tick
        vessel_snapshots, port_snapshots = self._env.snapshot_list["vessels"], self._env.snapshot_list["ports"]
        port_idx, vessel_idx = event.port_idx, event.vessel_idx
        ticks = [max(0, tick - rt) for rt in range(state_shaping_conf["look_back"] - 1)]
        future_port_list = vessel_snapshots[tick: vessel_idx: 'future_stop_list'].astype('int')
        state = np.concatenate([
            port_snapshots[ticks: [port_idx] + list(future_port_list): port_attributes],
            vessel_snapshots[tick: vessel_idx: vessel_attributes]
        ])
        return state, {port_idx: state}

    def _translate_to_env_action(
        self, action_dict: Dict[Any, Union[np.ndarray, List[object]]], event: DecisionEvent,
    ) -> Dict[Any, object]:
        action_space = action_shaping_conf["action_space"]
        finite_vsl_space = action_shaping_conf["finite_vessel_space"]
        has_early_discharge = action_shaping_conf["has_early_discharge"]

        port_idx, model_action = list(action_dict.items()).pop()

        vsl_idx, action_scope = event.vessel_idx, event.action_scope
        vsl_snapshots = self._env.snapshot_list["vessels"]
        vsl_space = vsl_snapshots[self._env.tick:vsl_idx:vessel_attributes][2] if finite_vsl_space else float("inf")

        percent = abs(action_space[model_action[0]])
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

    def _get_reward(self, env_action_dict: Dict[Any, object], event: DecisionEvent, tick: int) -> Dict[Any, float]:
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

    def _post_step(self, cache_element: CacheElement) -> None:
        self._info["env_metric"] = self._env.metrics

    def _post_eval_step(self, cache_element: CacheElement) -> None:
        self._post_step(cache_element)

    def post_collect(self, info_list: list, ep: int) -> None:
        # print the env metric from each rollout worker
        for info in info_list:
            print(f"env summary (episode {ep}): {info['env_metric']}")

        # print the average env metric
        if len(info_list) > 1:
            metric_keys, num_envs = info_list[0]["env_metric"].keys(), len(info_list)
            avg_metric = {key: sum(info["env_metric"][key] for info in info_list) / num_envs for key in metric_keys}
            print(f"average env summary (episode {ep}): {avg_metric}")

    def post_evaluate(self, info_list: list, ep: int) -> None:
        self.post_collect(info_list, ep)


class MyActorNet(DiscreteACBasedNet):
    def __init__(self, state_dim: int, action_num: int) -> None:
        super(MyActorNet, self).__init__(state_dim=state_dim, action_num=action_num)
        self._actor = FullyConnected(input_dim=state_dim, output_dim=action_num, **actor_net_conf)
        self._optim = Adam(self._actor.parameters(), lr=actor_learning_rate)

    def _get_action_probs_impl(self, states: torch.Tensor) -> torch.Tensor:
        return self._actor(states)


class MyCriticNet(VNet):
    def __init__(self, state_dim: int) -> None:
        super(MyCriticNet, self).__init__(state_dim=state_dim)
        self._critic = FullyConnected(input_dim=state_dim, **critic_net_conf)
        self._optim = RMSprop(self._critic.parameters(), lr=critic_learning_rate)

    def _get_v_values(self, states: torch.Tensor) -> torch.Tensor:
        return self._critic(states).squeeze(-1)


def get_trpo_trainer(state_dim: int, name: str) -> TRPOTrainer:
    return TRPOTrainer(
        name=name,
        reward_discount=.0,
        params=TRPOParams(
            get_v_critic_net_func=lambda: MyCriticNet(state_dim),
            grad_iters=10,
            critic_loss_cls=torch.nn.SmoothL1Loss,
            min_logp=None,
            lam=.0,
            clip_ratio=0.1,
        ),
    )


def get_ppo_trainer(state_dim: int, name: str) -> PPOTrainer:
    return PPOTrainer(
        name=name,
        reward_discount=.0,
        params=PPOParams(
            get_v_critic_net_func=lambda: MyCriticNet(state_dim),
            grad_iters=10,
            critic_loss_cls=torch.nn.SmoothL1Loss,
            min_logp=None,
            lam=.0,
            clip_ratio=0.1,
        ),
    )


# env and shaping config
reward_shaping_conf = {
    "time_window": 99,
    "fulfillment_factor": 1.0,
    "shortage_factor": 1.0,
    "time_decay": 0.97,
}
state_shaping_conf = {
    "look_back": 7,
    "max_ports_downstream": 2,
}
port_attributes = ["empty", "full", "on_shipper", "on_consignee", "booking", "shortage", "fulfillment"]
vessel_attributes = ["empty", "full", "remaining_space"]
action_shaping_conf = {
    "action_space": [(i - 10) / 10 for i in range(21)],
    "finite_vessel_space": True,
    "has_early_discharge": True,
}
state_dim = (
    (state_shaping_conf["look_back"] + 1) * (state_shaping_conf["max_ports_downstream"] + 1) * len(port_attributes)
    + len(vessel_attributes)
)
action_num = len(action_shaping_conf["action_space"])

actor_net_conf = {
    "hidden_dims": [256, 128, 64],
    "activation": torch.nn.Tanh,
    "softmax": True,
    "batch_norm": False,
    "head": True,
}
critic_net_conf = {
    "hidden_dims": [256, 128, 64],
    "output_dim": 1,
    "activation": torch.nn.LeakyReLU,
    "softmax": False,
    "batch_norm": True,
    "head": True,
}

actor_learning_rate = 0.001
critic_learning_rate = 0.001

learn_env = Env(scenario="cim", topology="toy.4p_ssdd_l0.0", durations=500)
test_env = learn_env
num_agents = len(learn_env.agent_idx_list)
agent2policy = {agent: f"ppo_{agent}.policy" for agent in learn_env.agent_idx_list}
policies = [DiscretePolicyGradient(name=f"ppo_{i}.policy", policy_net=MyActorNet(state_dim, action_num)) for i in
            range(num_agents)]
trainers = [get_trpo_trainer(state_dim, f"ppo_{i}") for i in range(num_agents)]

rl_component_bundle = RLComponentBundle(
    env_sampler=CIMEnvSampler(
        learn_env=learn_env,
        test_env=test_env,
        policies=policies,
        agent2policy=agent2policy,
        reward_eval_delay=reward_shaping_conf["time_window"],
    ),
    agent2policy=agent2policy,
    policies=policies,
    trainers=trainers,
)

env_sampler = rl_component_bundle.env_sampler

num_episodes = 30
eval_schedule = [5, 10, 15, 20, 25, 30]
eval_point_index = 0

training_manager = TrainingManager(rl_component_bundle=rl_component_bundle)

# main loop
for ep in range(1, num_episodes + 1):
    result = env_sampler.sample()
    experiences: List[List[ExpElement]] = result["experiences"]
    info_list: List[dict] = result["info"]

    print("Collecting result:")
    env_sampler.post_collect(info_list, ep)
    print()

    training_manager.record_experiences(experiences)
    training_manager.train_step()

    if ep == eval_schedule[eval_point_index]:
        eval_point_index += 1
        result = env_sampler.eval()

        print("Evaluation result:")
        env_sampler.post_evaluate(result["info"], ep)
        print()

training_manager.exit()
