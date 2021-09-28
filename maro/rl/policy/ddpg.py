# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from typing import Callable, List, Tuple, Union

import numpy as np
import torch

from maro.communication import SessionMessage
from maro.rl.exploration import gaussian_noise
from maro.rl.modeling import ContinuousACNet
from maro.rl.utils import MsgKey, MsgTag, average_grads
from maro.utils import clone

from .policy import RLPolicy
from .replay import ReplayMemory


class DDPG(RLPolicy):
    """The Deep Deterministic Policy Gradient (DDPG) algorithm.

    References:
        https://arxiv.org/pdf/1509.02971.pdf
        https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ddpg

    Args:
        name (str): Unique identifier for the policy.
        ac_net (ContinuousACNet): DDPG policy and q-value models.
        reward_discount (float): Reward decay as defined in standard RL terminology.
        num_epochs (int): Number of training epochs per call to ``learn``. Defaults to 1.
        update_target_every (int): Number of training rounds between policy target model updates.
        q_value_loss_cls: A string indicating a loss class provided by torch.nn or a custom loss class for
            the Q-value loss. If it is a string, it must be a key in ``TORCH_LOSS``. Defaults to "mse".
        q_value_loss_coeff (float): Coefficient for policy loss in the total loss function, e.g.,
            loss = policy_loss + ``q_value_loss_coeff`` * q_value_loss. Defaults to 1.0.
        soft_update_coeff (float): Soft update coefficient, e.g., target_model = (soft_update_coeff) * eval_model +
            (1-soft_update_coeff) * target_model. Defaults to 1.0.
        exploration_strategy (Tuple[Callable, dict]): A 2-tuple that consists of a) a function that takes a state
            (single or batch), an action (single or batch), the total number of possible actions and a set of keyword
            arguments, and returns an exploratory action (single or batch depending on the input), and b) a dictionary
            of keyword arguments for the function in a) (this will be assigned to the ``exploration_params`` member
            variable). Defaults to (``gaussian_noise``, {"mean": .0, "stddev": 1.0, "relative": False}).
        exploration_scheduling_option (List[tuple]): A list of 3-tuples specifying the exploration schedulers to be
            registered to the exploration parameters. Each tuple consists of an exploration parameter name, an
            exploration scheduler class (subclass of ``AbsExplorationScheduler``) and keyword arguments for that class.
            The exploration parameter name must be a key in the keyword arguments (second element) of
            ``exploration_strategy``. Defaults to an empty list.
        replay_memory_capacity (int): Capacity of the replay memory. Defaults to 10000.
        exploration_params (dict): Keyword arguments for ``exploration_func``. Defaults to {"mean": .0, "stddev": 1.0,
            "relative": False}.
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
        ac_net: ContinuousACNet,
        reward_discount: float,
        num_epochs: int = 1,
        update_target_every: int = 5,
        q_value_loss_cls="mse",
        q_value_loss_coeff: float = 1.0,
        soft_update_coeff: float = 1.0,
        exploration_strategy: Tuple[Callable, dict] = (gaussian_noise, {"mean": .0, "stddev": 1.0, "relative": False}),
        exploration_scheduling_options: List[tuple] = [],
        replay_memory_capacity: int = 1000000,
        random_overwrite: bool = False,
        warmup: int = 50000,
        rollout_batch_size: int = 1000,
        train_batch_size: int = 32,
        device: str = None
    ):
        if not isinstance(ac_net, ContinuousACNet):
            raise TypeError("model must be an instance of 'ContinuousACNet'")

        if any(opt[0] not in exploration_strategy[1] for opt in exploration_scheduling_options):
            raise ValueError(
                f"The first element of an exploration scheduling option must be one of "
                f"{list(exploration_strategy[1].keys())}"
            )

        super().__init__(name)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.ac_net = ac_net.to(self.device)
        self.target_ac_net = clone(self.ac_net)
        self.target_ac_net.eval()
        self.reward_discount = reward_discount
        self.num_epochs = num_epochs
        self.update_target_every = update_target_every
        self.q_value_loss_func = q_value_loss_cls()
        self.q_value_loss_coeff = q_value_loss_coeff
        self.soft_update_coeff = soft_update_coeff

        self._ac_net_version = 0
        self._target_ac_net_version = 0

        self._replay_memory = ReplayMemory(
            replay_memory_capacity, self.ac_net.input_dim, action_dim=1, random_overwrite=random_overwrite
        )
        self.warmup = warmup
        self.rollout_batch_size = rollout_batch_size
        self.train_batch_size = train_batch_size

        self.exploration_func = exploration_strategy[0]
        self._exploration_params = clone(exploration_strategy[1])
        self.exploration_schedulers = [
            opt[1](self._exploration_params, opt[0], **opt[2]) for opt in exploration_scheduling_options
        ]

    def __call__(self, states: np.ndarray):
        if self._replay_memory.size < self.warmup:
            return np.random.uniform(
                low=self.ac_net.out_min, high=self.ac_net.out_max,
                size=(states.shape[0] if len(states.shape) > 1 else 1, self.ac_net.action_dim)
            )

        self.ac_net.eval()
        states = torch.from_numpy(states).to(self.device)
        if len(states.shape) == 1:
            states = states.unsqueeze(dim=0)
        with torch.no_grad():
            actions = self.ac_net.get_action(states).cpu().numpy()

        if not self.greedy:
            actions = self.exploration_func(states, actions, **self._exploration_params)
        return actions

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

        self._replay_memory.put(
            np.expand_dims(state, axis=0),
            np.expand_dims(action, axis=0),
            np.expand_dims(reward, axis=0),
            np.expand_dims(next_state, axis=0),
            np.expand_dims(terminal, axis=0)
        )

    def get_rollout_info(self):
        """Randomly sample a batch of transitions from the replay memory.

        This is used in a distributed learning setting and the returned data will be sent to its parent instance
        on the learning side (serving as the source of the latest model parameters) for training.
        """
        return self._replay_memory.sample(self.rollout_batch_size)

    def get_batch_loss(self, batch: dict, explicit_grad: bool = False) -> dict:
        """Compute loss for a data batch.

        Args:
            batch (dict): A batch containing "states", "actions", "rewards", "next_states" and "terminals" as keys.
            explicit_grad (bool): If True, the gradients should be returned as part of the loss information. Defaults
                to False.
        """
        self.ac_net.train()
        states = torch.from_numpy(batch["states"]).to(self.device)
        next_states = torch.from_numpy(["next_states"]).to(self.device)
        actual_actions = torch.from_numpy(batch["actions"]).to(self.device)
        rewards = torch.from_numpy(batch["rewards"]).to(self.device)
        terminals = torch.from_numpy(batch["terminals"]).float().to(self.device)
        if len(actual_actions.shape) == 1:
            actual_actions = actual_actions.unsqueeze(dim=1)  # (N, 1)

        with torch.no_grad():
            next_q_values = self.target_ac_net.value(next_states)
        target_q_values = (rewards + self.reward_discount * (1 - terminals) * next_q_values).detach()  # (N,)

        # loss info
        loss_info = {}
        q_values = self.ac_net(states, actions=actual_actions).squeeze(dim=1)  # (N,)
        q_loss = self.q_value_loss_func(q_values, target_q_values)
        policy_loss = -self.ac_net.value(states).mean()
        loss = policy_loss + self.q_value_loss_coeff * q_loss
        loss_info = {
            "policy_loss": policy_loss.detach().cpu().numpy(),
            "q_loss": q_loss.detach().cpu().numpy(),
            "loss": loss.detach().cpu().numpy() if explicit_grad else loss
        }
        if explicit_grad:
            loss_info["grad"] = self.ac_net.get_gradients(loss)

        return loss_info

    def update(self, loss_info_list: List[dict]):
        """Update the model parameters with gradients computed by multiple gradient workers.

        Args:
            loss_info_list (List[dict]): A list of dictionaries containing loss information (including gradients)
                computed by multiple gradient workers.
        """
        self.ac_net.apply_gradients(average_grads([loss_info["grad"] for loss_info in loss_info_list]))
        if self._ac_net_version - self._target_ac_net_version == self.update_target_every:
            self._update_target()

    def learn(self, batch: dict):
        """Learn from a batch containing data required for policy improvement.

        Args:
            batch (dict): A batch containing "states", "actions", "rewards", "next_states" and "terminals" as keys.
        """
        self._replay_memory.put(
            batch["states"], batch["actions"], batch["rewards"], batch["next_states"], batch["terminals"]
        )
        self.improve()

    def improve(self):
        """Learn using data from the replay memory."""
        for _ in range(self.num_epochs):
            train_batch = self._replay_memory.sample(self.train_batch_size)
            self.ac_net.step(self.get_batch_loss(train_batch)["loss"])
            self._ac_net_version += 1
            if self._ac_net_version - self._target_ac_net_version == self.update_target_every:
                self._update_target()

    def learn_with_data_parallel(self, batch: dict):
        assert hasattr(self, '_proxy'), "learn_with_data_parallel is invalid before data_parallel is called."

        self._replay_memory.put(
            batch["states"], batch["actions"], batch["rewards"], batch["next_states"], batch["terminals"]
        )
        for _ in range(self.num_epochs):
            msg_dict = defaultdict(lambda: defaultdict(dict))
            worker_id_list = self.request_workers()
            for worker_id in worker_id_list:
                msg_dict[worker_id][MsgKey.GRAD_TASK][self._name] = self._replay_memory.sample(
                    self.train_batch_size // len(worker_id_list))
                msg_dict[worker_id][MsgKey.POLICY_STATE][self._name] = self.get_state()
                # data-parallel by multiple hosts/workers
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
                            loss_info_by_policy[policy_name].append(loss_info["grad"])
                        else:
                            raise TypeError(f"Wrong type of loss_info: {type(loss_info)}")
                    dones += 1
                    if dones == len(msg_dict):
                        break
            # build dummy computation graph by `get_batch_loss` before apply gradients.
            # batch_size=2 because torch.nn.functional.batch_norm doesn't support batch_size=1.
            _ = self.get_batch_loss(self._replay_memory.sample(2), explicit_grad=True)
            self.update(loss_info_by_policy[self._name])

    def _update_target(self):
        # soft-update target network
        self.target_ac_net.soft_update(self.ac_net, self.soft_update_coeff)
        self._target_ac_net_version = self._ac_net_version

    def exploration_step(self):
        for sch in self.exploration_schedulers:
            sch.step()

    def get_state(self):
        return self.ac_net.get_state()

    def set_state(self, state):
        self.ac_net.set_state(state)

    def load(self, path: str):
        """Load the policy state from disk."""
        checkpoint = torch.load(path)
        self.ac_net.set_state(checkpoint["ac_net"])
        self._ac_net_version = checkpoint["ac_net_version"]
        self.target_ac_net.set_state(checkpoint["target_ac_net"])
        self._target_ac_net_version = checkpoint["target_ac_net_version"]
        self._replay_memory = checkpoint["replay_memory"]

    def save(self, path: str):
        """Save the policy state to disk."""
        policy_state = {
            "ac_net": self.ac_net.get_state(),
            "ac_net_version": self._ac_net_version,
            "target_ac_net": self.target_ac_net.get_state(),
            "target_ac_net_version": self._target_ac_net_version,
            "replay_memory": self._replay_memory
        }
        torch.save(policy_state, path)
