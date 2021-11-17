# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import torch

from maro.rl.exploration import gaussian_noise
from maro.rl.modeling_v2 import DiscretePolicyGradientNetwork
from maro.rl.modeling_v2.critic_model import MultiDiscreteQCriticNetwork
from maro.rl.policy_v2.policy_base import MultiRLPolicy
from maro.rl.policy_v2.policy_interfaces import MultiDiscreteActionMixin
from maro.rl.policy_v2.replay import ReplayMemory
from maro.rl.utils import average_grads
from maro.utils import clone


class MultiDiscreteActorCritic(MultiDiscreteActionMixin, MultiRLPolicy):
    """
    References:
        MADDPG paper: https://arxiv.org/pdf/1706.02275.pdf

    Args:
        name (str): Unique identifier for the policy.
        agent_nets (List[DiscretePolicyGradientNetwork]): Networks for all sub-agents.
        critic_net (MultiDiscreteQCriticNetwork): Critic's network.
        reward_discount (float): Reward decay as defined in standard RL terminology.
        shared_state_dim (int): State dim of the shared part of state. Defaults to 0.
        num_epochs (int): Number of training epochs per call to ``learn``. Defaults to 1.
        update_target_every (int): Number of training rounds between policy target model updates.
        min_logp (float): Lower bound for clamping logP values during learning. This is to prevent logP from becoming
            very large in magnitude and causing stability issues. Defaults to None, which means no lower bound.
        critic_loss_cls: A string indicating a loss class provided by torch.nn or a custom loss class for computing
            the critic loss. If it is a string, it must be a key in ``TORCH_LOSS``. Defaults to "mse".
        critic_loss_coef (float): Coefficient of critic loss.
        soft_update_coef (float): Soft update coeficient, e.g., target_model = (soft_update_coef) * eval_model +
            (1-soft_update_coef) * target_model. Defaults to 1.0.
        clip_ratio (float): Clip ratio in the PPO algorithm (https://arxiv.org/pdf/1707.06347.pdf). Defaults to None,
            in which case the actor loss is calculated using the usual policy gradient theorem.
        lam (float): Lambda value for generalized advantage estimation (TD-Lambda). Defaults to 0.9.
        replay_memory_capacity (int): Capacity of the replay memory. Defaults to 10000.
        random_overwrite (bool): This specifies overwrite behavior when the replay memory capacity is reached. If True,
            overwrite positions will be selected randomly. Otherwise, overwrites will occur sequentially with
            wrap-around. Defaults to False.
        rollout_batch_size (int): Size of the experience batch to use as roll-out information by calling
            ``get_rollout_info``. Defaults to 1000.
        train_batch_size (int): Batch size for training the Q-net. Defaults to 32.
        device (str): Identifier for the torch device. The ``ac_net`` will be moved to the specified device. If it is
            None, the device will be set to "cpu" if cuda is unavailable and "cuda" otherwise. Defaults to None.
    """
    def __init__(
        self,
        name: str,
        agent_nets: List[DiscretePolicyGradientNetwork],
        critic_net: MultiDiscreteQCriticNetwork,
        reward_discount: float,
        shared_state_dim: int = 0,
        num_epochs: int = 1,
        update_target_every: int = 5,
        min_logp: float = None,
        critic_loss_cls: Callable = None,
        critic_loss_coef: float = 1.0,
        soft_update_coef: float = 1.0,
        exploration_strategy: Tuple[Callable, dict] = (gaussian_noise, {"mean": .0, "stddev": 1.0, "relative": False}),
        exploration_scheduling_options: List[tuple] = [],
        clip_ratio: float = None,
        lam: float = 0.9,
        replay_memory_capacity: int = 1000000,
        random_overwrite: bool = False,
        warmup: int = 50000,
        rollout_batch_size: int = 1000,
        train_batch_size: int = 32,
        device: str = None
    ) -> None:
        if any(opt[0] not in exploration_strategy[1] for opt in exploration_scheduling_options):
            raise ValueError(
                f"The first element of an exploration scheduling option must be one of "
                f"{list(exploration_strategy[1].keys())}"
            )

        super(MultiDiscreteActorCritic, self).__init__(name=name, device=device)

        self._critic_net = critic_net
        self._agent_nets = agent_nets
        self._shared_state_dim = shared_state_dim
        self._num_sub_agents = len(self._agent_nets)
        self._local_state_dims = [net.state_dim - self._shared_state_dim for net in self._agent_nets]
        self._total_action_dim = sum([net.action_dim for net in self._agent_nets])
        self._total_state_dim = self._critic_net.state_dim - self._total_action_dim

        # for each agent, individual state = local state + shared state
        assert all(dim >= 0 for dim in self._local_state_dims)
        assert self._total_state_dim == sum(self._local_state_dims) + self._shared_state_dim

        # target network
        self._target_critic_net = clone(critic_net)
        self._target_critic_net.eval()
        self._target_agent_nets = [clone(agent_net) for agent_net in self._agent_nets]
        for agent_net in self._target_agent_nets:
            agent_net.eval()

        self._reward_discount = reward_discount
        self.num_epochs = num_epochs
        self.update_target_every = update_target_every
        self._min_logp = min_logp
        self._critic_loss_func = critic_loss_cls() if critic_loss_cls is not None else torch.nn.MSELoss()
        self._critic_loss_coef = critic_loss_coef
        self._soft_update_coef = soft_update_coef
        self._clip_ratio = clip_ratio
        self._lam = lam

        self._critic_net_version = 0
        self._target_net_version = 0

        # List of single agent replay memory for multi-agent scenario.
        self._replay_memory = [ReplayMemory(
            replay_memory_capacity, self._local_state_dims[i], action_dim=1, random_overwrite=random_overwrite)
            for i in range(self._num_sub_agents)]
        self.warmup = warmup
        self.rollout_batch_size = rollout_batch_size
        self.train_batch_size = train_batch_size

        self.exploration_func = exploration_strategy[0]
        self._exploration_params = clone(exploration_strategy[1])
        self.exploration_schedulers = [
            opt[1](self._exploration_params, opt[0], **opt[2]) for opt in exploration_scheduling_options
        ]

    def _get_action_nums(self) -> List[int]:
        return [net.action_num for net in self._agent_nets]

    def _get_action_dims(self) -> List[int]:
        return [net.action_dim for net in self._agent_nets]

    def _get_state_dim(self) -> int:
        return self._critic_net.state_dim

    def _get_agent_num(self) -> int:
        return self._num_sub_agents

    def _call_impl(self, states: List[np.ndarray], agent_ids: List[int]) -> Iterable:
        actions = self.get_actions(states, agent_ids)
        return [{
                "action": action,  # [num_sub_agent]
                } for action in actions]

    def _get_global_states(self, state_list: List[np.ndarray]) -> np.ndarray:
        """Get concatenated global state of all sub-agents. Encode the individual states into global state.

        Args:
            state_list (List[np.ndarray]): List of local states encoded with shared state.

        Returns:
            global_states (np.ndarray): Global state with shape [batch_size, sum(local_state_dim) + shared_state_dim]
        """
        shared_state = state_list[0][:, self._total_state_dim - self._shared_state_dim:]
        global_state_list = []
        for local_state_dim, state in zip(self._local_state_dims, state_list):
            local_state = state[:, :local_state_dim - self._shared_state_dim]
            global_state_list.append(local_state)
        global_state_list.append(shared_state)
        return np.concatenate(global_state_list, axis=1)

    def _get_actions_impl(self, input_states: List[np.ndarray], agent_ids: List[int]) -> List[np.ndarray]:
        """

        Args:
            input_states (List[np.ndarray]): global state with shape [batch_size, total_state_dim]
            agent_ids (List[int]): the corresponding agent id of input states.

        Returns:
            actions (List): [num_sub_agent, action_dim]
            logps (List): [num_sub_agent, 1]
        """
        for net in self._agent_nets:
            net.eval()

        with torch.no_grad():
            actions = []
            for agent_id, state in zip(agent_ids, input_states):  # iterate `num_sub_agent` times
                net = self._agent_nets[agent_id]
                state = torch.from_numpy(state).unsqueeze(0).to(self._device)
                action, _ = net.get_actions_and_logps(state, self._exploring)  # [batch_size], [batch_size]
                actions.append(action)
        return actions

    def _get_values_by_states_and_actions(self, states: torch.Tensor, actions: List[torch.Tensor]) -> torch.Tensor:
        """Get Q-values by all agents' states and actions.

        Args:
            states (torch.Tensor): States of all sub-agents, with shape [batch_size, total_state_dim].
            actions (List[torch.Tensor]): List of sub-agents' action, with shape [batch_size, action_num].

        Returns:
            q_values (torch.Tensor): The Q-values with shape [batch_size].
        """
        action_tensor = torch.stack(actions).T  # [batch_size, sub_agent_num]
        return self._critic_net.q_critic(states, action_tensor)

    def record(
        self, key: str, state: np.ndarray, action: dict, reward: float,
        next_state: np.ndarray, terminal: bool
    ) -> None:
        """Record experience information of a single agent. The info would be saved by the key(agent index).
        """
        if next_state is None:
            next_state = np.zeros(state.shape, dtype=np.float32)

        agent_idx = int(key)
        action = action["action"]  # keys: action, logp
        self._replay_memory[agent_idx].put(
            np.expand_dims(state, axis=0),
            np.expand_dims(action, axis=0),
            np.expand_dims(reward, axis=0),
            np.expand_dims(next_state, axis=0),
            np.expand_dims(terminal, axis=0)
        )

    def get_rollout_info(self) -> dict:
        """Randomly sample a batch of transitions from all agents' replay memories.

        This is used in a distributed learning setting and the returned data will be sent to its parent instance
        on the learning side (serving as the source of the latest model parameters) for training.
        """
        indexes = np.random.choice(self._replay_memory[0].size, size=self.rollout_batch_size)
        rollout_infos = [self._replay_memory[i].sample_index(indexes) for i in range(self._num_sub_agents)]
        return rollout_infos

    def get_batch_loss(self, batch: List[Dict[str, np.ndarray]], explicit_grad: bool = False) -> dict:
        """Compute loss for all subagents.

        Args:
            batch (List[Dict[np.ndarray]]): List of experience data (SARS) batch collected from subagents.
                For each subagent's batch(dict), required keys: states, actions, rewards, next_states, terminals.
                Each of them is shaped: [batch_size, individual_state_dim/action_dim/...].
            explicit_grad (bool): Whether explicitly return gradients. Defaults to False.

        Returns:
            loss_info (Dict[torch.Tensor]): The information of losses. Required keys: critic_loss, actor_loss, loss.
                Optional keys: actor_grads, critic_grad.
        """
        for i, net in enumerate(self._agent_nets):
            net.train()
        self._critic_net.train()

        # batch formatting
        states = self._get_global_states([sub_batch["states"] for sub_batch in batch])
        next_states = self._get_global_states([sub_batch["next_states"] for sub_batch in batch])
        # `sum` reward of subagents for coorperative mode
        rewards = np.concatenate([sub_batch["rewards"].reshape(-1, 1) for sub_batch in batch], axis=1).sum(axis=1)
        terminals = np.all(
            np.concatenate([sub_batch["terminals"].reshape(-1, 1) for sub_batch in batch], axis=1), axis=1)

        # type converting
        states = torch.from_numpy(states).to(self._device)  # [batch_size, total_state_dim]
        next_states = torch.from_numpy(next_states).to(self._device)  # [batch_size, total_state_dim]
        actions = [torch.from_numpy(sub_batch["actions"]).to(self._device).long() for sub_batch in batch]

        rewards = torch.from_numpy(rewards).to(self._device)  # [batch_size]
        terminals = torch.from_numpy(terminals).float().to(self._device)  # [batch_size]

        # critic loss
        with torch.no_grad():
            next_actions = [agent.get_actions_and_logps(
                                torch.from_numpy(sub_batch["next_states"]).to(self._device), self._exploring)[0]
                            for agent, sub_batch in zip(self._target_agent_nets, batch)]
            next_action_tensor = torch.stack(next_actions).T  # [batch_size, sub_agent_num]
            next_q_values = self._target_critic_net.q_critic(states, next_action_tensor)
        target_q_values = (rewards + self._reward_discount * (1 - terminals) * next_q_values).detach()  # [batch_size]
        q_values = self._get_values_by_states_and_actions(states, actions)  # [batch_size]
        critic_loss = self._critic_loss_func(q_values, target_q_values)

        # actor losses
        actor_losses = []
        for i in range(self._num_sub_agents):
            net = self._agent_nets[i]
            state = torch.from_numpy(batch[i]["states"]).to(self._device)
            new_action, logp = net.get_actions_and_logps(state, self._exploring)  # [batch_size], [batch_size]
            cur_actions = [action for action in actions]
            cur_actions[i] = new_action
            q_values = self._get_values_by_states_and_actions(states, cur_actions)
            actor_loss = -(q_values * logp).mean()
            actor_losses.append(actor_loss)

        # total loss
        loss = sum(actor_losses) + self._critic_loss_coef * critic_loss

        loss_info = {
            "critic_loss": critic_loss,
            "actor_losses": actor_losses,
            "loss": loss.detach().cpu().numpy() if explicit_grad else loss
        }

        if explicit_grad:
            loss_info["actor_grads"] = [net.get_gradients(loss) for net in self._agent_nets]
            loss_info["critic_grad"] = self._critic_net.get_gradients(loss)

        return loss_info

    def update(self, loss_info_list: List[dict]) -> None:
        for i, net in enumerate(self._agent_nets):
            net.apply_gradients(average_grads([loss_info["actor_grads"][i] for loss_info in loss_info_list]))
        self._critic_net.apply_gradients(average_grads([loss_info["critic_grad"] for loss_info in loss_info_list]))
        self._critic_net_version += 1
        if self._critic_net_version - self._target_net_version == self.update_target_every:
            self._update_target()

    def learn(self, batch: List[dict]) -> None:
        """Learn from a batch of experience data.

        Args:
            batch (List[dict]): A list of experience data batch collected from all sub-agents.
        """
        for i, sub_batch in enumerate(batch):
            self._replay_memory[i].put(
                sub_batch["states"], sub_batch["actions"], sub_batch["rewards"],
                sub_batch["next_states"], sub_batch["terminals"]
            )
        self.improve()

    def improve(self) -> None:
        for _ in range(self.num_epochs):
            indexes = np.random.choice(self._replay_memory[0].size, size=self.train_batch_size)
            train_batch = [memory.sample_index(indexes) for memory in self._replay_memory]
            loss = self.get_batch_loss(train_batch)
            for net, actor_loss in zip(self._agent_nets, loss["actor_losses"]):
                net.step(actor_loss)
            self._critic_net.step(loss["critic_loss"])
            self._critic_net_version += 1
            if self._critic_net_version - self._target_net_version == self.update_target_every:
                self._update_target()

    def _update_target(self):
        # soft-update target network
        self._target_critic_net.soft_update(self._critic_net, self._soft_update_coef)
        for target_agent, agent in zip(self._target_agent_nets, self._agent_nets):
            target_agent.soft_update(agent, self._soft_update_coef)
        self._target_net_version = self._critic_net_version

    def get_exploration_params(self):
        return clone(self._exploration_params)

    def exploration_step(self):
        for sch in self.exploration_schedulers:
            sch.step()

    def get_state(self) -> object:
        return {
            "agent_nets": [net.get_state() for net in self._agent_nets],
            "critic_net": self._critic_net.get_state()
        }

    def set_state(self, policy_state: object) -> None:
        assert isinstance(policy_state, object), f"Expected `object` but got `{type(policy_state)}` instead."
        for net, state in zip(self._agent_nets, policy_state["agent_nets"]):
            net.set_state(state)
        self._critic_net.set_state(policy_state["critic_net"])

    def load(self, path: str) -> None:
        checkpoint = torch.load(path)
        self._critic_net.set_state(checkpoint["critic_net"])
        self._critic_net_version = checkpoint["critic_net_version"]
        self._target_critic_net.set_state(checkpoint["target_critic_net"])
        self._target_net_version = checkpoint["target_net_version"]
        self._replay_memory = checkpoint["replay_memory"]

        for net, ckpt in zip(self._agent_nets, checkpoint["agent_nets"]):
            net.set_state(ckpt)

        for net, ckpt in zip(self._target_agent_nets, checkpoint["target_agent_nets"]):
            net.set_state(ckpt)

    def save(self, path: str) -> None:
        """Save the policy state to disk."""
        net_states = self.get_state()
        policy_state = {
            "critic_net": net_states["critic_net"],
            "agent_nets": net_states["agent_nets"],
            "critic_net_version": self._critic_net_version,
            "target_critic_net": self._target_critic_net.get_state(),
            "target_agent_nets": [agent.get_state() for agent in self._target_agent_nets],
            "target_net_version": self._target_net_version,
            "replay_memory": self._replay_memory
        }
        torch.save(policy_state, path)
