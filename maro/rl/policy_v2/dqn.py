# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from typing import Callable, Iterable, List, Tuple, Union

import numpy as np
import torch

from maro.communication import SessionMessage
from maro.rl.exploration import epsilon_greedy
from maro.rl.modeling_v2 import DiscreteQNetwork
from maro.rl.utils import MsgKey, MsgTag, average_grads
from maro.utils import clone

from .policy_base import RLPolicyV2
from .policy_interfaces import DiscreteQNetworkMixin
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
        max_priority (float): Maximum priority value to use for new experiences. Defaults to 1e8.
    """
    def __init__(
        self,
        replay_memory: ReplayMemory,
        *,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_step: float = 0.001,
        max_priority: float = 1e8
    ) -> None:
        if beta > 1.0:
            raise ValueError("beta should be between 0.0 and 1.0")
        self._replay_memory = replay_memory
        self._sum_tree = np.zeros(2 * self._replay_memory.capacity - 1)
        self._alpha = alpha
        self._beta = beta
        self._beta_step = beta_step
        self._eps = 1e-7
        self._max_priority = max_priority

    def total(self) -> float:
        """Return the sum of priorities over all experiences."""
        return self._sum_tree[0]

    def set_max_priority(self, indexes: List[int]) -> None:
        """Set the priorities of newly added experiences to the maximum value."""
        self.update(indexes, [self._max_priority] * len(indexes))

    def update(self, indexes: List[int], td_errors: List[float]) -> None:
        """Update priority values at given indexes."""
        for idx, err in zip(indexes, td_errors):
            priority = self._get_priority(err)
            tree_idx = idx + self._replay_memory.capacity - 1
            delta = priority - self._sum_tree[tree_idx]
            self._sum_tree[tree_idx] = priority
            self._update(tree_idx, delta)

    def sample(self, size: int) -> Tuple[List[int], float]:
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

        self._beta = min(1., self._beta + self._beta_step)
        sampling_probabilities = np.array(priorities) / (self.total() + 1e-8)
        is_weights = np.power(self._replay_memory.size * sampling_probabilities, -self._beta)
        is_weights /= (is_weights.max() + 1e-8)

        return indexes, is_weights

    def _get_priority(self, error: Union[torch.Tensor, np.ndarray, float]) -> float:
        if isinstance(error, torch.Tensor):
            error = error.detach().numpy()
        return (np.abs(error) + self._eps) ** self._alpha

    def _update(self, idx: int, delta: float) -> None:
        """Propagate priority change all the way to the root node."""
        parent = (idx - 1) // 2
        self._sum_tree[parent] += delta
        if parent != 0:
            self._update(parent, delta)

    def _get(self, idx: int, sampled_val: float) -> int:
        """Get a leaf node according to a randomly sampled value."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self._sum_tree):
            return idx

        if sampled_val <= self._sum_tree[left]:
            return self._get(left, sampled_val)
        else:
            return self._get(right, sampled_val - self._sum_tree[left])


class DQN(DiscreteQNetworkMixin, RLPolicyV2):
    """The Deep-Q-Networks algorithm.

    See https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf for details.

    Args:
        name (str): Unique identifier for the policy.
        q_net (DiscreteQNet): Q-value model.
        reward_discount (float): Reward decay as defined in standard RL terminology.
        num_epochs (int): Number of training epochs per call to ``learn``. Defaults to 1.
        update_target_every (int): Number of gradient steps between target model updates.
        soft_update_coef (float): Soft update coefficient, e.g.,
            target_model = (soft_update_coef) * eval_model + (1-soft_update_coef) * target_model.
            Defaults to 1.0.
        double (bool): If True, the next Q values will be computed according to the double DQN algorithm,
            i.e., q_next = Q_target(s, argmax(Q_eval(s, a))). Otherwise, q_next = max(Q_target(s, a)).
            See https://arxiv.org/pdf/1509.06461.pdf for details. Defaults to False.
        exploration_strategy (Tuple[Callable, dict]): A 2-tuple that consists of a) a function that takes a state
            (single or batch), an action (single or batch), the total number of possible actions and a set of keyword
            arguments, and returns an exploratory action (single or batch depending on the input), and b) a dictionary
            of keyword arguments for the function in a) (this will be assigned to the ``_exploration_params`` member
            variable). Defaults to (``epsilon_greedy``, {"epsilon": 0.1}).
        exploration_scheduling_options (List[tuple]): A list of 3-tuples specifying the exploration schedulers to be
            registered to the exploration parameters. Each tuple consists of an exploration parameter name, an
            exploration scheduler class (subclass of ``AbsExplorationScheduler``) and keyword arguments for that class.
            The exploration parameter name must be a key in the keyword arguments (second element) of
            ``exploration_strategy``. Defaults to an empty list.
        replay_memory_capacity (int): Capacity of the replay memory. Defaults to 1000000.
        random_overwrite (bool): This specifies overwrite behavior when the replay memory capacity is reached. If True,
            overwrite positions will be selected randomly. Otherwise, overwrites will occur sequentially with
            wrap-around. Defaults to False.
        warmup (int): When the total number of experiences in the replay memory is below this threshold,
            ``choose_action`` will return uniformly random actions for warm-up purposes. Defaults to 50000.
        rollout_batch_size (int): Size of the experience batch to use as roll-out information by calling
            ``get_rollout_info``. Defaults to 1000.
        train_batch_size (int): Batch size for training the Q-net. Defaults to 32.
        prioritized_replay_kwargs (dict): Keyword arguments for prioritized experience replay. See
            ``PrioritizedExperienceReplay`` for details. Defaults to None, in which case experiences will be sampled
            from the replay memory uniformly randomly.
        device (str): Identifier for the torch device. The ``q_net`` will be moved to the specified device. If it is
            None, the device will be set to "cpu" if cuda is unavailable and "cuda" otherwise. Defaults to None.
    """
    def __init__(
        self,
        name: str,
        q_net: DiscreteQNetwork,
        reward_discount: float = 0.9,
        num_epochs: int = 1,
        update_target_every: int = 5,
        soft_update_coef: float = 0.1,
        double: bool = False,
        exploration_strategy: Tuple[Callable, dict] = (epsilon_greedy, {"epsilon": 0.1}),
        exploration_scheduling_options: List[tuple] = None,
        replay_memory_capacity: int = 1000000,
        random_overwrite: bool = False,
        warmup: int = 50000,
        rollout_batch_size: int = 1000,
        train_batch_size: int = 32,
        prioritized_replay_kwargs: dict = None,
        device: str = None
    ):
        super(DQN, self).__init__(name=name, device=device)

        if exploration_scheduling_options is None:
            exploration_scheduling_options = []

        if not isinstance(q_net, DiscreteQNetwork):
            raise TypeError("Model must be an instance of 'DiscreteQNetwork'")

        if any(opt[0] not in exploration_strategy[1] for opt in exploration_scheduling_options):
            raise ValueError(
                f"The first element of an exploration scheduling option must be one of "
                f"{list(exploration_strategy[1].keys())}"
            )

        self._q_net = q_net.to(self._device)
        self._target_q_net: DiscreteQNetwork = clone(q_net)
        self._target_q_net.eval()
        self._q_net_version = 0
        self._target_q_net_version = 0

        self._num_actions = self._q_net.action_num
        self._reward_discount = reward_discount
        self._num_epochs = num_epochs
        self._update_target_every = update_target_every
        self._soft_update_coef = soft_update_coef
        self._double = double

        self._replay_memory = ReplayMemory(
            replay_memory_capacity, self._q_net.state_dim,
            self._q_net.action_dim, random_overwrite=random_overwrite
        )
        self._warmup = warmup
        self._rollout_batch_size = rollout_batch_size
        self._train_batch_size = train_batch_size
        self._prioritized_replay = prioritized_replay_kwargs is not None
        if self._prioritized_replay:
            self._per = PrioritizedExperienceReplay(self._replay_memory, **prioritized_replay_kwargs)
        else:
            self._loss_func = torch.nn.MSELoss()

        self._exploration_func = exploration_strategy[0]
        self._exploration_params = clone(exploration_strategy[1])  # deep copy is needed to avoid unwanted sharing
        self._exploration_schedulers = [
            opt[1](self._exploration_params, opt[0], **opt[2]) for opt in exploration_scheduling_options
        ]

    def _call_impl(self, states: np.ndarray) -> Iterable:
        if self._replay_memory.size < self._warmup:
            return np.random.randint(self._num_actions, size=(states.shape[0] if len(states.shape) > 1 else 1,))

        self._q_net.eval()
        states: torch.Tensor = torch.from_numpy(states).to(self._device)
        if len(states.shape) == 1:
            states = states.unsqueeze(dim=0)
        with torch.no_grad():
            actions = self._q_net.get_actions(states, exploring=False)  # [batch_size, 1]
            actions = actions.squeeze(-1)  # [batch_size]

        if not self._exploring:
            return actions.cpu().numpy()
        else:
            return self._exploration_func(states, actions.cpu().numpy(), self._num_actions, **self._exploration_params)

    def data_parallel(self, *args, **kwargs) -> None:
        raise NotImplementedError  # TODO

    def _get_q_values_for_all_actions(self, states: np.ndarray) -> np.ndarray:
        return self._q_net.q_values_for_all_actions(torch.Tensor(states)).numpy()

    def _get_q_values(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        q_matrix = self.q_values_for_all_actions(states)  # [batch_size, action_num]
        return np.take_along_axis(q_matrix, actions, axis=1)

    def _get_action_num(self) -> int:
        return self._q_net.action_num

    def _get_state_dim(self) -> int:
        return self._q_net.state_dim

    def record(
        self,
        key: str,
        state: np.ndarray,
        action: Union[int, float, np.ndarray],
        reward: float,
        next_state: np.ndarray,
        terminal: bool
    ) -> None:
        if next_state is None:
            next_state = np.zeros(state.shape, dtype=np.float32)

        indexes = self._replay_memory.put(
            np.expand_dims(state, axis=0),
            np.expand_dims(action, axis=0),
            np.expand_dims(reward, axis=0),
            np.expand_dims(next_state, axis=0),
            np.expand_dims(terminal, axis=0)
        )
        if self._prioritized_replay:
            self._per.set_max_priority(indexes)

    def get_rollout_info(self) -> dict:
        """Randomly sample a batch of transitions from the replay memory.

        This is used in a distributed learning setting and the returned data will be sent to its parent instance
        on the learning side (serving as the source of the latest model parameters) for training.
        """
        return self._replay_memory.sample(self._rollout_batch_size)

    def _get_batch(self, batch_size: int = None) -> dict:
        if batch_size is None:
            batch_size = self._train_batch_size
        if self._prioritized_replay:
            indexes, is_weights = self._per.sample(batch_size)
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
            return self._replay_memory.sample(self._train_batch_size)

    def get_batch_loss(self, batch: dict, explicit_grad: bool = False) -> dict:
        """Compute loss for a data batch.

        Args:
            batch (dict): A batch containing "states", "actions", "rewards", "next_states" and "terminals" as keys.
            explicit_grad (bool): If True, the gradients should be returned as part of the loss information. Defaults
                to False.
        """
        self._q_net.train()
        states: torch.Tensor = torch.from_numpy(batch["states"]).to(self._device)
        next_states: torch.Tensor = torch.from_numpy(batch["next_states"]).to(self._device)
        actions: torch.Tensor = torch.from_numpy(batch["actions"]).to(self._device)
        rewards: torch.Tensor = torch.from_numpy(batch["rewards"]).to(self._device)
        terminals: torch.Tensor = torch.from_numpy(batch["terminals"]).float().to(self._device)

        # get target Q values
        with torch.no_grad():
            if self._double:
                actions_by_eval_q_net = self._q_net.get_actions(next_states, exploring=False)
                next_q_values = self._target_q_net.q_values(next_states, actions_by_eval_q_net)
            else:
                actions = self._target_q_net.get_actions(next_states, exploring=False)
                next_q_values = self._target_q_net.q_values(next_states, actions)

        target_q_values = (rewards + self._reward_discount * (1 - terminals) * next_q_values).detach()

        # loss info
        loss_info = {}
        q_values = self._q_net.q_values(states, actions)
        td_errors = target_q_values - q_values
        if self._prioritized_replay:
            is_weights = torch.from_numpy(batch["is_weights"]).to(self._device)
            loss = (td_errors * is_weights).mean()
            loss_info["td_errors"], loss_info["indexes"] = td_errors.detach().cpu().numpy(), batch["indexes"]
        else:
            loss = self._loss_func(q_values, target_q_values)

        loss_info["loss"] = loss.detach().cpu().numpy() if explicit_grad else loss
        if explicit_grad:
            loss_info["grad"] = self._q_net.get_gradients(loss)
        return loss_info

    def update(self, loss_info_list: List[dict]) -> None:
        """Update the Q-net parameters with gradients computed by multiple gradient workers.

        Args:
            loss_info_list (List[dict]): A list of dictionaries containing loss information (including gradients)
                computed by multiple gradient workers.
        """
        if self._prioritized_replay:
            for loss_info in loss_info_list:
                self._per.update(loss_info["indexes"], loss_info["td_errors"])

        self._q_net.apply_gradients(average_grads([loss_info["grad"] for loss_info in loss_info_list]))
        self._q_net_version += 1
        # soft-update target network
        if self._q_net_version - self._target_q_net_version == self._update_target_every:
            self._update_target()

    def learn(self, batch: dict) -> None:
        """Learn from a batch containing data required for policy improvement.

        Args:
            batch (dict): A batch containing "states", "actions", "rewards", "next_states" and "terminals" as keys.
        """
        self._replay_memory.put(
            batch["states"], batch["actions"], batch["rewards"], batch["next_states"], batch["terminals"]
        )
        self.improve()

    def improve(self) -> None:
        """Learn using data from the replay memory."""
        for _ in range(self._num_epochs):
            loss_info = self.get_batch_loss(self._get_batch())
            if self._prioritized_replay:
                self._per.update(loss_info["indexes"], loss_info["td_errors"])
            self._q_net.step(loss_info["loss"])
            self._q_net_version += 1
            if self._q_net_version - self._target_q_net_version == self._update_target_every:
                self._update_target()

    def _update_target(self) -> None:
        self._target_q_net.soft_update(self._q_net, self._soft_update_coef)
        self._new_target_q_net_version = self._q_net_version

    def learn_with_data_parallel(self, batch: dict, worker_id_list: list) -> None:
        assert hasattr(self, '_proxy'), "learn_with_data_parallel is invalid before data_parallel is called."

        self._replay_memory.put(
            batch["states"], batch["actions"], batch["rewards"], batch["next_states"], batch["terminals"]
        )
        for _ in range(self._num_epochs):
            msg_dict = defaultdict(lambda: defaultdict(dict))
            for worker_id in worker_id_list:
                msg_dict[worker_id][MsgKey.GRAD_TASK][self._name] = self._get_batch(
                    self._train_batch_size // len(worker_id_list))
                msg_dict[worker_id][MsgKey.POLICY_STATE][self._name] = self.get_state()
                # data-parallel by multiple remote gradient workers
                self._proxy.isend(SessionMessage(
                    MsgTag.COMPUTE_GRAD, self._proxy.name, worker_id, body=msg_dict[worker_id]))
            dones = 0
            loss_info_by_policy = {self._name: []}
            for msg in self._proxy.receive():
                if msg.tag == MsgTag.COMPUTE_GRAD_DONE:
                    for policy_name, loss_info in msg.body[MsgKey.LOSS_INFO].items():
                        if isinstance(loss_info, list):
                            loss_info_by_policy[policy_name] += loss_info
                        elif isinstance(loss_info, dict):
                            loss_info_by_policy[policy_name].append(loss_info)
                        else:
                            raise TypeError(f"Wrong type of loss_info: {type(loss_info)}")
                    dones += 1
                    if dones == len(msg_dict):
                        break
            # build dummy computation graph before apply gradients.
            _ = self.get_batch_loss(self._get_batch(), explicit_grad=True)
            self.update(loss_info_by_policy[self._name])

    def get_exploration_params(self):
        return clone(self._exploration_params)

    def exploration_step(self) -> None:
        """Update the exploration parameters according to the exploration scheduler."""
        for sch in self._exploration_schedulers:
            sch.step()

    def get_state(self) -> object:
        return self._q_net.get_state()

    def set_state(self, policy_state: object) -> None:
        self._q_net.set_state(policy_state)

    def load(self, path: str) -> None:
        """Load the policy state from disk."""
        checkpoint = torch.load(path)
        self._q_net.set_state(checkpoint["q_net"])
        self._q_net_version = checkpoint["q_net_version"]
        self._target_q_net.set_state(checkpoint["target_q_net"])
        self._target_q_net_version = checkpoint["target_q_net_version"]
        self._replay_memory = checkpoint["replay_memory"]
        if self._prioritized_replay:
            self._per = checkpoint["prioritized_replay"]

    def save(self, path: str) -> None:
        """Save the policy state to disk."""
        policy_state = {
            "q_net": self._q_net.get_state(),
            "q_net_version": self._q_net_version,
            "target_q_net": self._target_q_net.get_state(),
            "target_q_net_version": self._target_q_net_version,
            "replay_memory": self._replay_memory
        }
        if self._prioritized_replay:
            policy_state["prioritized_replay"] = self._per
        torch.save(policy_state, path)
