# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Union

import numpy as np
import torch

from maro.rl.exploration import DiscreteSpaceExploration, EpsilonGreedyExploration
from maro.rl.modeling import DiscreteQNet
from maro.utils.exception.rl_toolkit_exception import InvalidExperience

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
        replay_memory: ReplayMemory,
        *,
        batch_size: int = 32,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_step: float = 0.001,
        max_priority: float = 1e8
    ):
        if beta > 1.0:
            raise ValueError("beta should be between 0.0 and 1.0")
        self._replay_memory = replay_memory
        self._sum_tree = np.zeros(2 * self._replay_memory.capacity - 1)
        self.batch_size = batch_size
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

    def sample(self):
        """Priority-based sampling."""
        indexes, priorities = [], []
        segment_len = self.total() / self.batch_size
        for i in range(self.batch_size):
            low, high = segment_len * i, segment_len * (i + 1)
            sampled_val = np.random.uniform(low=low, high=high)
            idx = self._get(0, sampled_val)
            data_idx = idx - self._replay_memory.capacity + 1
            indexes.append(data_idx)
            priorities.append(self._sum_tree[idx])

        self.beta = min(1., self.beta + self.beta_step)
        sampling_probabilities = priorities / self.total()
        is_weights = np.power(self._replay_memory.size * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

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
    """

    class Batch(RLPolicy.Batch):
        """Wrapper for a set of experiences.

        An experience consists of state, action, reward, next state.
        """
        __slots__ = ["states", "actions", "rewards", "next_states", "terminal"]

        def __init__(
            self,
            states: np.ndarray,
            actions: np.ndarray,
            rewards: np.ndarray,
            next_states: np.ndarray,
            terminal: np.ndarray,
            indexes: list = None,
            is_weights: list = None
        ):
            if not len(states) == len(actions) == len(rewards) == len(next_states) == len(terminal):
                raise InvalidExperience("values of contents should consist of lists of the same length")
            super().__init__()
            self.states = states
            self.actions = actions
            self.rewards = rewards
            self.next_states = next_states
            self.terminal = terminal
            self.is_weights = is_weights
            self.indexes = indexes

        @property
        def size(self):
            return len(self.states)


    class LossInfo(RLPolicy.LossInfo):

        __slots__ = ["td_errors", "indexes"]

        def __init__(self, loss, td_errors, indexes, grad=None):
            super().__init__(loss, grad)
            self.loss = loss
            self.td_errors = td_errors
            self.indexes = indexes
            self.grad = grad


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
        prioritized_replay_kwargs: dict = None,
        remote: bool = False
    ):
        if not isinstance(q_net, DiscreteQNet):
            raise TypeError("model must be an instance of 'DiscreteQNet'")

        super().__init__(name, remote=remote)
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

        self._replay_memory = ReplayMemory(self.Batch, replay_memory_capacity, random_overwrite=random_overwrite)
        self.prioritized_replay = prioritized_replay_kwargs is not None
        if self.prioritized_replay:
            self._per = PrioritizedExperienceReplay(self._replay_memory, **prioritized_replay_kwargs)
        else:
            self._loss_func = torch.nn.MSELoss()

        self.exploration = exploration
        self.exploring = True  # set initial exploration status to True

    def choose_action(self, states) -> Union[int, np.ndarray]:
        self.q_net.eval()
        with torch.no_grad():
            q_for_all_actions = self.q_net(states)  # (batch_size, num_actions)
            _, actions = q_for_all_actions.max(dim=1)

        actions = actions.cpu().numpy()
        if self.exploring:
            if self.exploration.action_space is None:
                self.exploration.set_action_space(np.arange(q_for_all_actions.shape[1]))
            actions = self.exploration(actions, state=states)
        return actions[0] if len(actions) == 1 else actions

    def _put_in_replay_memory(self, traj: dict):
        batch = self.Batch(traj["states"][:-1], traj["actions"][:-1], traj["rewards"], traj["states"][1:])
        indexes = self._replay_memory.put(batch)
        if self.prioritized_replay:
            self._per.set_max_priority(indexes)

    def _sample(self):
        if self.prioritized_replay:
            indexes, is_weights = self._per.sample()
        else:
            indexes = np.random.choice(self._replay_memory.size)
            is_weights = None

        return self.Batch(
            [self._replay_memory.data["states"][idx] for idx in indexes],
            [self._replay_memory.data["actions"][idx] for idx in indexes],
            [self._replay_memory.data["rewards"][idx] for idx in indexes],
            [self._replay_memory.data["next_states"][idx] for idx in indexes],
            indexes=indexes,
            is_weights=is_weights
        )

    def get_batch_loss(self, batch: Batch, explicit_grad: bool = False):
        assert self.q_net.trainable, "q_net needs to have at least one optimizer registered."
        self.q_net.train()
        states, next_states = batch.states, batch.next_states
        actions = torch.from_numpy(np.asarray(batch.actions)).to(self.device)
        rewards = torch.from_numpy(np.asarray(batch.rewards)).to(self.device)

        # get target Q values
        with torch.no_grad():
            if self.double:
                actions_by_eval_q_net = self.q_net.get_action(next_states)[0]
                next_q_values = self.target_q_net.q_values(next_states, actions_by_eval_q_net)
            else:
                next_q_values = self.target_q_net.get_action(next_states)[1]  # (N,)

        target_q_values = (rewards + self.reward_discount * (1 - ) * next_q_values).detach()  # (N,)

        # gradient step
        q_values = self.q_net.q_values(states, actions)
        td_errors = target_q_values - q_values
        if self.prioritized_replay:
            is_weights = torch.from_numpy(np.asarray(batch.is_weights)).to(self.device)
            loss = (td_errors * is_weights).mean()
        else:
            loss = self._loss_func(q_values, target_q_values)

        grad = self.q_net.get_gradients(loss) if explicit_grad else None
        return self.LossInfo(loss, td_errors, batch.indexes, grad=grad)

    def update_with_multi_loss_info(self, loss_info_list: List[LossInfo]):
        if self.prioritized_replay:
            for loss_info in loss_info_list:
                self._per.update(loss_info.indexes, loss_info.td_errors)

        self.q_net.apply_gradients([loss_info.grad for loss_info in loss_info_list])
        self._q_net_version += 1
        if self._q_net_version - self._target_q_net_version == self.update_target_every:
            self._update_target()

    def learn_from_multi_trajectories(self, trajectories: List[dict]):
        for traj in trajectories:
            self._put_in_replay_memory(traj)

        if self.remote:
            # TODO: distributed grad computation
            pass
        else:
            for _ in range(self.num_epochs):
                loss_info = self.get_batch_loss(self._sample())
                self.q_net.step(loss_info.loss)
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
        self.exploring = False

    def explore(self):
        self.exploring = True

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
