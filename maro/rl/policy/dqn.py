# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Union

import numpy as np
import torch

from maro.rl.exploration import DiscreteSpaceExploration, EpsilonGreedyExploration
from maro.rl.modeling import DiscreteQNet

from .policy import RLPolicy
from .replay import ReplayMemory


class PrioritizedExperienceReplay:
    """Prioritized Experience Replay (PER).

    References:
        https://arxiv.org/pdf/1511.05952.pdf
        https://github.com/rlcode/per

    The implementation here is based on direct proportional prioritization (the first variant in the paper).
    The rank-based variant is not implemented here.

    Args:
        replay_memory (ReplayMemory): experience manager the sampler is associated with.
        alpha (float): Prioritization strength. Sampling probabilities are calculated according to
            P = p_i^alpha / sum(p_k^alpha). Defaults to 0.6.
        beta (float): Bias annealing strength using weighted importance sampling (IS) techniques.
            IS weights are calculated according to (N * P)^(-beta), where P is the sampling probability.
            This value of ``beta`` should not exceed 1.0, which corresponds to full annealing. Defaults to 0.4.
        beta_step (float): The amount ``beta`` is incremented by after each get() call until it reaches 1.0.
            Defaults to 0.001.
    """
    def __init__(
        self,
        replay_memory: ReplayMemory,
        *,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_step: float = 0.001,
        max_priority: float = 1e8
    ):
        if beta > 1.0:
            raise ValueError("beta should be between 0.0 and 1.0")
        self._replay_memory = replay_memory
        self._sum_tree = np.zeros(2 * self._replay_memory.capacity - 1)
        self.alpha = alpha
        self.beta = beta
        self.beta_step = beta_step
        self.eps = 1e-7
        self._max_priority = max_priority

    def total(self):
        """Return the sum of priorities over all experiences."""
        return self._sum_tree[0]

    def set_max_priority(self, indexes):
        """Set the priorities of newly added experiences to the maximum value."""
        self.update(indexes, [self._max_priority] * len(indexes))

    def update(self, indexes, td_errors):
        """Update priority values at given indexes."""
        for idx, err in zip(indexes, td_errors):
            priority = self._get_priority(err)
            tree_idx = idx + self._replay_memory.capacity - 1
            delta = priority - self._sum_tree[tree_idx]
            self._sum_tree[tree_idx] = priority
            self._update(tree_idx, delta)

    def sample(self, size: int):
        """Priority-based sampling."""
        indexes, priorities = [], []
        segment_len = self.total() / size
        for i in range(size):
            low, high = segment_len * i, segment_len * (i + 1)
            sampled_val = np.random.uniform(low=low, high=high)
            idx = self._get(0, sampled_val)
            data_idx = idx - self._replay_memory.capacity + 1
            indexes.append(data_idx)
            priorities.append(self._sum_tree[idx])

        self.beta = min(1., self.beta + self.beta_step)
        sampling_probabilities = priorities / (self.total() + 1e-8)
        is_weights = np.power(self._replay_memory.size * sampling_probabilities, -self.beta)
        is_weights /= (is_weights.max() + 1e-8)

        return indexes, is_weights

    def _get_priority(self, error):
        return (np.abs(error) + self.eps) ** self.alpha

    def _update(self, idx, delta):
        """Propagate priority change all the way to the root node."""
        parent = (idx - 1) // 2
        self._sum_tree[parent] += delta
        if parent != 0:
            self._update(parent, delta)

    def _get(self, idx, sampled_val):
        """Get a leaf node according to a randomly sampled value."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self._sum_tree):
            return idx

        if sampled_val <= self._sum_tree[left]:
            return self._get(left, sampled_val)
        else:
            return self._get(right, sampled_val - self._sum_tree[left])


class DQN(RLPolicy):
    """The Deep-Q-Networks algorithm.

    See https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf for details.

    Args:
        name (str): Unique identifier for the policy.
        q_net (DiscreteQNet): Q-value model.
        reward_discount (float): Reward decay as defined in standard RL terminology.
        train_epochs (int): Number of training epochs per call to ``update()``. Defaults to 1.
        update_target_every (int): Number of gradient steps between target model updates.
        soft_update_coeff (float): Soft update coefficient, e.g.,
            target_model = (soft_update_coeff) * eval_model + (1-soft_update_coeff) * target_model.
            Defaults to 1.0.
        double (bool): If True, the next Q values will be computed according to the double DQN algorithm,
            i.e., q_next = Q_target(s, argmax(Q_eval(s, a))). Otherwise, q_next = max(Q_target(s, a)).
            See https://arxiv.org/pdf/1509.06461.pdf for details. Defaults to False.
        exploration (DiscreteSpaceExploration): Exploration strategy for generating exploratory actions. Defaults to
            ``EpsilonGreedyExploration``.
        replay_memory_capacity (int): Capacity of the replay memory. Defaults to 10000.
        random_overwrite (bool): This specifies overwrite behavior when the replay memory capacity is reached. If True,
            overwrite positions will be selected randomly. Otherwise, overwrites will occur sequentially with
            wrap-around. Defaults to False.
        batch_size (int): Training sample. Defaults to 32.
    """

    def __init__(
        self,
        name: str,
        q_net: DiscreteQNet,
        reward_discount: float = 0.9,
        num_epochs: int = 1,
        update_target_every: int = 5,
        soft_update_coeff: float = 0.1,
        double: bool = False,
        exploration: DiscreteSpaceExploration = EpsilonGreedyExploration(),
        replay_memory_capacity: int = 10000,
        random_overwrite: bool = False,
        train_batch_size: int = 32,
        rollout_batch_size: int = 1000,
        prioritized_replay_kwargs: dict = None
    ):
        if not isinstance(q_net, DiscreteQNet):
            raise TypeError("model must be an instance of 'DiscreteQNet'")

        super().__init__(name)
        self.q_net = q_net
        self.device = self.q_net.device
        if self.q_net.trainable:
            self.target_q_net = q_net.copy()
            self.target_q_net.eval()
        else:
            self.target_q_net = None
        self._q_net_version = 0
        self._target_q_net_version = 0

        self.reward_discount = reward_discount
        self.num_epochs = num_epochs
        self.update_target_every = update_target_every
        self.soft_update_coeff = soft_update_coeff
        self.double = double

        self._replay_memory = ReplayMemory(
            replay_memory_capacity, self.q_net.input_dim, action_dim=1, random_overwrite=random_overwrite
        )
        self.rollout_batch_size = rollout_batch_size
        self.train_batch_size = train_batch_size
        self.prioritized_replay = prioritized_replay_kwargs is not None
        if self.prioritized_replay:
            self._per = PrioritizedExperienceReplay(self._replay_memory, **prioritized_replay_kwargs)
        else:
            self._loss_func = torch.nn.MSELoss()

        self.exploration = exploration
        self.greedy = True  # set initial exploration status to True

    def choose_action(self, states: np.ndarray) -> Union[int, np.ndarray]:
        self.q_net.eval()
        states = torch.from_numpy(states).to(self.device)
        if len(states.shape) == 1:
            states = states.unsqueeze(dim=0)
        with torch.no_grad():
            q_for_all_actions = self.q_net(states)  # (batch_size, num_actions)
            _, actions = q_for_all_actions.max(dim=1)

        actions = actions.cpu().numpy()
        if not self.greedy:
            if self.exploration.action_space is None:
                self.exploration.set_action_space(np.arange(q_for_all_actions.shape[1]))
            actions = self.exploration(actions, state=states)
        return actions[0] if len(actions) == 1 else actions

    def record(
        self,
        key: str,
        state: np.ndarray,
        action: Union[int, float, np.ndarray],
        reward: float,
        next_state: np.ndarray,
        terminal: bool
    ):
        if next_state is None:
            next_state = np.zeros(state.shape, dtype=np.float32)

        indexes = self._replay_memory.put(
            np.expand_dims(state, axis=0),
            np.expand_dims(action, axis=0),
            np.expand_dims(reward, axis=0),
            np.expand_dims(next_state, axis=0),
            np.expand_dims(terminal, axis=0)
        )
        if self.prioritized_replay:
            self._per.set_max_priority(indexes)

    def get_rollout_info(self):
        return self._replay_memory.sample(self.rollout_batch_size)

    def _get_batch(self):
        if self.prioritized_replay:
            indexes, is_weights = self._per.sample(self.train_batch_size)
            return {
                "states": self._replay_memory.states[indexes],
                "actions": self._replay_memory.actions[indexes],
                "rewards": self._replay_memory.rewards[indexes],
                "next_states": self._replay_memory.next_states[indexes],
                "terminals": self._replay_memory.terminals[indexes],
                "indexes": indexes,
                "is_weights": is_weights
            }
        else:
            return self._replay_memory.sample(self.train_batch_size)

    def get_batch_loss(self, batch: dict, explicit_grad: bool = False):
        assert self.q_net.trainable, "q_net needs to have at least one optimizer registered."
        self.q_net.train()
        states = torch.from_numpy(batch["states"]).to(self.device)
        next_states = torch.from_numpy(batch["next_states"]).to(self.device)
        actions = torch.from_numpy(batch["actions"]).to(self.device)
        rewards = torch.from_numpy(batch["rewards"]).to(self.device)
        terminals = torch.from_numpy(batch["terminals"]).float().to(self.device)

        # get target Q values
        with torch.no_grad():
            if self.double:
                actions_by_eval_q_net = self.q_net.get_action(next_states)[0]
                next_q_values = self.target_q_net.q_values(next_states, actions_by_eval_q_net)
            else:
                next_q_values = self.target_q_net.get_action(next_states)[1]  # (N,)

        target_q_values = (rewards + self.reward_discount * (1 - terminals) * next_q_values).detach()  # (N,)

        # loss info
        loss_info = {}
        q_values = self.q_net.q_values(states, actions)
        # print(f"target: {target_q_values}, eval: {q_values}")
        td_errors = target_q_values - q_values
        if self.prioritized_replay:
            is_weights = torch.from_numpy(batch["is_weights"]).to(self.device)
            loss = (td_errors * is_weights).mean()
            loss_info["td_errors"], loss_info["indexes"] = td_errors.detach().cpu().numpy(), batch["indexes"]
        else:
            loss = self._loss_func(q_values, target_q_values)

        loss_info["loss"] = loss
        if explicit_grad:
            loss_info["grad"] = self.q_net.get_gradients(loss)
        return loss_info

    def update(self, loss_info_list: List[dict]):
        if self.prioritized_replay:
            for loss_info in loss_info_list:
                self._per.update(loss_info["indexes"], loss_info["td_errors"])

        self.q_net.apply_gradients([loss_info["grad"] for loss_info in loss_info_list])
        self._q_net_version += 1
        if self._q_net_version - self._target_q_net_version == self.update_target_every:
            self._update_target()

    def learn(self, batch: dict):
        self._replay_memory.put(
            batch["states"], batch["actions"], batch["rewards"], batch["next_states"], batch["terminals"]
        )

        for _ in range(self.num_epochs):
            loss_info = self.get_batch_loss(self._get_batch())
            if self.prioritized_replay:
                self._per.update(loss_info["indexes"], loss_info["td_errors"])
            self.q_net.step(loss_info["loss"])
            self._q_net_version += 1
            if self._q_net_version - self._target_q_net_version == self.update_target_every:
                self._update_target()

    def _update_target(self):
        # soft-update target network
        self.target_q_net.soft_update(self.q_net, self.soft_update_coeff)
        self._target_q_net_version = self._q_net_version

    @property
    def exploration_params(self):
        return self.exploration.parameters

    def exploit(self):
        self.greedy = True

    def explore(self):
        self.greedy = False

    def exploration_step(self):
        self.exploration.step()

    def set_state(self, policy_state):
        self.q_net.load_state_dict(policy_state["eval"])
        if "target" in policy_state:
            self.target_q_net.load_state_dict(policy_state["target"])

    def get_state(self, inference: bool = True):
        policy_state = {"eval": self.q_net.state_dict()}
        if not inference and self.target_q_net:
            policy_state["target"] = self.target_q_net.state_dict()
        return policy_state
