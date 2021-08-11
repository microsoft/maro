# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple, Union

import numpy as np
import torch

from maro.rl.algorithms.abs_algorithm import AbsPolicy
from maro.rl.replay import ExperienceSet, ReplayMemory
from maro.rl.exploration import DiscreteSpaceExploration, EpsilonGreedyExploration
from maro.rl.model import DiscreteQNet


class PrioritizedSampler:
    """Sampler for Prioritized Experience Replay (PER).

    References:
        https://arxiv.org/pdf/1511.05952.pdf
        https://github.com/rlcode/per

    The implementation here is based on direct proportional prioritization (the first variant in the paper).
    The rank-based variant is not implemented here.

    Args:
        replay_memory (ReplayMemory): experience manager the sampler is associated with.
        batch_size (int): mini-batch size. Defaults to 32.
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
        memory_capacity: int,
        batch_size: int = 32,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_step: float = 0.001
    ):
        if beta > 1.0:
            raise ValueError("beta should be between 0.0 and 1.0")
        self.memory_capacity = memory_capacity
        self._sum_tree = np.zeros(2 * self.memory_capacity - 1)
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.beta_step = beta_step
        self.eps = 1e-7
        self._max_priority = 1e8

    def total(self):
        """Return the sum of priorities over all experiences."""
        return self._sum_tree[0]

    def on_new(self, experience_set: ExperienceSet, indexes: List[int]):
        """Set the priorities of newly added experiences to the maximum value."""
        self.update(indexes, [self._max_priority] * len(indexes))

    def update(self, indexes, td_errors):
        """Update priority values at given indexes."""
        for idx, err in zip(indexes, td_errors):
            priority = self._get_priority(err)
            tree_idx = idx + self.memory_capacity - 1
            delta = priority - self._sum_tree[tree_idx]
            self._sum_tree[tree_idx] = priority
            self._update(tree_idx, delta)

    def get(self):
        """Priority-based sampling."""
        indexes, priorities = [], []
        segment_len = self.total() / self.batch_size
        for i in range(self.batch_size):
            low, high = segment_len * i, segment_len * (i + 1)
            sampled_val = np.random.uniform(low=low, high=high)
            idx = self._get(0, sampled_val)
            data_idx = idx - self.memory_capacity + 1
            indexes.append(data_idx)
            priorities.append(self._sum_tree[idx])

        self.beta = min(1., self.beta + self.beta_step)
        sampling_probabilities = priorities / self.total()
        is_weights = np.power(self.replay_memory.size * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        return indexes, is_weights
        return ExperienceSet(
            states=[self.replay_memory.data["states"][idx] for idx in indexes],
            actions=[self.replay_memory.data["actions"][idx] for idx in indexes],
            rewards=[self.replay_memory.data["rewards"][idx] for idx in indexes],
            next_states=[self.replay_memory.data["next_states"][idx] for idx in indexes],
            info=[{"index": idx, "is_weight": wt} for idx, wt in zip(indexes, is_weights)]
        )

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


class DQN(AbsPolicy):
    """The Deep-Q-Networks algorithm.

    See https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf for details.

    Args:
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
            an ``EpsilonGreedyExploration`` instance.
    """
    def __init__(
        self,
        q_net: DiscreteQNet,
        reward_discount: float = 0.9,
        train_epochs: int = 1,
        update_target_every: int = 5,
        soft_update_coeff: float = 0.1,
        double: bool = False,
        prioritized_experience_sampler: PrioritizedSampler = None,
        replay_memory_capacity: int = 10000,
        random_overwrite: bool = False,
        exploration: DiscreteSpaceExploration = EpsilonGreedyExploration()
    ):
        if not isinstance(q_net, DiscreteQNet):
            raise TypeError("model must be an instance of 'DiscreteQNet'")

        super().__init__(exploration=exploration)
        self.q_net = q_net
        if self.q_net.trainable:
            self.target_q_net = q_net.copy()
            self.target_q_net.eval()
        else:
            self.target_q_net = None
        self.reward_discount = reward_discount
        self.train_epochs = train_epochs
        self.update_target_every = update_target_every
        self.soft_update_coeff = soft_update_coeff
        self.double = double
        self.prioritized_experience_sampler = prioritized_experience_sampler

        self.device = self.q_net.device
        if not self.prioritized_experience_replay:
            self._loss_func = torch.nn.MSELoss()
        self._replay_memory = ReplayMemory(replay_memory_capacity, random_overwrite=random_overwrite) 
        self._q_net_version = 0
        self._target_q_net_version = 0

    def choose_action(self, states, explore: bool = True) -> Union[int, np.ndarray]:
        self.q_net.eval()
        with torch.no_grad():
            q_for_all_actions = self.q_net(states)  # (batch_size, num_actions)
            _, actions = q_for_all_actions.max(dim=1)

        actions = actions.cpu().numpy()
        if explore:
            if self.exploration.action_space is None:
                self.exploration.set_action_space(np.arange(q_for_all_actions.shape[1]))
            actions = self.exploration(actions, state=states)
        return actions[0] if len(actions) == 1 else actions

    def apply(self, grad_dict: dict):
        self.q_net.apply(grad_dict)
        self._q_net_version += 1
        if self._q_net_version - self._target_q_net_version == self.update_target_every:
            self._update_target()

    def learn(self, exp: ExperienceSet):
        assert self.q_net.trainable, "q_net needs to have at least one optimizer registered."
        self.q_net.train()
        self._replay_memory.put(exp)
        for _ in range(self.train_epochs):
            # sample from the replay memory
            experience_set = self.sampler.get()
            states, next_states = experience_set.states, experience_set.next_states
            actions = torch.from_numpy(np.asarray(experience_set.actions)).to(self.device)
            rewards = torch.from_numpy(np.asarray(experience_set.rewards)).to(self.device)
            if self.prioritized_experience_replay:
                indexes = [info["index"] for info in experience_set.info]
                is_weights = torch.tensor([info["is_weight"] for info in experience_set.info]).to(self.device)

            # get target Q values
            with torch.no_grad():
                if self.double:
                    actions_by_eval_q_net = self.q_net.get_action(next_states)[0]
                    next_q_values = self.target_q_net.q_values(next_states, actions_by_eval_q_net)
                else:
                    next_q_values = self.target_q_net.get_action(next_states)[1]  # (N,)

            target_q_values = (rewards + self.reward_discount * next_q_values).detach()  # (N,)

            # gradient step
            q_values = self.q_net.q_values(states, actions)
            if self.prioritized_experience_replay:
                td_errors = target_q_values - q_values
                loss = (td_errors * is_weights).mean()
                self.sampler.update(indexes, td_errors.detach().cpu().numpy())
            else:
                loss = self._loss_func(q_values, target_q_values)
            self.q_net.step(loss)

        self._q_net_version += 1
        if self._q_net_version - self._target_q_net_version == self.update_target_every:
            self._update_target()

    def _update_target(self):
        # soft-update target network
        self.target_q_net.soft_update(self.q_net, self.soft_update_coeff)
        self._target_q_net_version = self._q_net_version

    def set_state(self, policy_state):
        self.q_net.load_state_dict(policy_state["eval"])
        if "target" in policy_state:
            self.target_q_net.load_state_dict(policy_state["target"])

    def get_state(self, inference: bool = True):
        policy_state = {"eval": self.q_net.state_dict()}
        if not inference and self.target_q_net:
            policy_state["target"] = self.target_q_net.state_dict()
        return policy_state
