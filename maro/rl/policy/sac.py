# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Union

import numpy as np
import torch

from maro.rl.modeling import ContinuousSACNet
from maro.rl.utils import average_grads
from maro.utils import clone

from .policy import RLPolicy
from .replay import ReplayMemory


class SoftActorCritic(RLPolicy):
    """The Soft Actor-Critic (SAC) algorithm.

    References:
        https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac

    Args:
        name (str): Unique identifier for the policy.
        sac_net (ContinuousSACNet): DDPG policy and q-value models.
        reward_discount (float): Reward decay as defined in standard RL terminology.
        num_epochs (int): Number of training epochs per call to ``learn``. Defaults to 1.
        update_target_every (int): Number of training rounds between policy target model updates.
        q_value_loss_cls: A string indicating a loss class provided by torch.nn or a custom loss class for
            the Q-value loss. If it is a string, it must be a key in ``TORCH_LOSS``. Defaults to "mse".
        q_value_loss_coeff (float): Coefficient for policy loss in the total loss function, e.g.,
            loss = policy_loss + ``q_value_loss_coeff`` * q_value_loss. Defaults to 1.0.
        soft_update_coeff (float): Soft update coefficient, e.g., target_model = (soft_update_coeff) * eval_model +
            (1-soft_update_coeff) * target_model. Defaults to 1.0.
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
        sac_net: ContinuousSACNet,
        reward_discount: float,
        alpha: float = 0.2,
        num_epochs: int = 1,
        update_target_every: int = 5,
        q_value_loss_cls=torch.nn.MSELoss,
        q_value_loss_coeff: float = 1.0,
        soft_update_coeff: float = 1.0,
        replay_memory_capacity: int = 1000000,
        random_overwrite: bool = False,
        warmup: int = 50000,
        rollout_batch_size: int = 1000,
        train_batch_size: int = 32,
        device: str = None
    ):
        if not isinstance(sac_net, ContinuousSACNet):
            raise TypeError("model must be an instance of 'ContinuousACNet'")

        super().__init__(name)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.sac_net = sac_net.to(self.device)
        self.target_sac_net = clone(self.sac_net)
        self.target_sac_net.eval()
        self.reward_discount = reward_discount
        self.alpha = alpha
        self.num_epochs = num_epochs
        self.update_target_every = update_target_every
        self.q_value_loss_func = q_value_loss_cls()
        self.q_value_loss_coeff = q_value_loss_coeff
        self.soft_update_coeff = soft_update_coeff

        self._sac_net_version = 0
        self._target_sac_net_version = 0

        self._replay_memory = ReplayMemory(
            replay_memory_capacity, self.sac_net.input_dim,
            action_dim=self.sac_net.action_dim, random_overwrite=random_overwrite
        )
        self.warmup = warmup
        self.rollout_batch_size = rollout_batch_size
        self.train_batch_size = train_batch_size

    def __call__(self, states: np.ndarray):
        if self._replay_memory.size < self.warmup:
            return np.random.uniform(
                low=self.sac_net.action_min, high=self.sac_net.action_max,
                size=(states.shape[0] if len(states.shape) > 1 else 1, self.sac_net.action_dim)
            )

        self.sac_net.eval()
        states = torch.from_numpy(states).to(self.device)
        if len(states.shape) == 1:
            states = states.unsqueeze(dim=0)
        with torch.no_grad():
            action = self.sac_net.get_action(states, deterministic=self.greedy).cpu().numpy()
        return action

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

    def get_batch_loss(self, batch: dict, explicit_grad: bool = False):
        raise NotImplementedError

    def update(self, loss_info_list: List[dict]):
        """Update the model parameters with gradients computed by multiple gradient workers.

        Args:
            loss_info_list (List[dict]): A list of dictionaries containing loss information (including gradients)
                computed by multiple gradient workers.
        """
        self.sac_net.apply_gradients(average_grads([loss_info["grad"] for loss_info in loss_info_list]))
        if self._sac_net_version - self._target_sac_net_version == self.update_target_every:
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

            ####################################################################
            states = torch.from_numpy(train_batch["states"]).to(self.device)
            next_states = torch.from_numpy(train_batch["next_states"]).to(self.device)
            actual_actions = torch.from_numpy(train_batch["actions"]).to(self.device)
            rewards = torch.from_numpy(train_batch["rewards"]).to(self.device)
            terminals = torch.from_numpy(train_batch["terminals"]).float().to(self.device)
            if len(actual_actions.shape) == 1:
                actual_actions = actual_actions.unsqueeze(dim=1)  # (N, 1)

            self.sac_net.train()

            # Update Q1 & Q2
            self.sac_net.q_optim.zero_grad()

            q1 = self.sac_net.get_q1_values(states, actual_actions)
            q2 = self.sac_net.get_q2_values(states, actual_actions)

            with torch.no_grad():
                next_actions, next_logps = self.sac_net(next_states, deterministic=False)
                target_next_q = torch.min(
                    self.target_sac_net.get_q1_values(next_states, next_actions),
                    self.target_sac_net.get_q2_values(next_states, next_actions)
                )
                q_target = rewards + self.reward_discount * (1 - terminals) * (target_next_q - self.alpha * next_logps)

            q_loss = self.q_value_loss_func(q1, q_target) + self.q_value_loss_func(q2, q_target)

            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.sac_net.q_params, 5)
            self.sac_net.q_optim.step()

            # Update policy
            for param in self.sac_net.q_params:
                param.requires_grad = False

            self.sac_net.policy_optim.zero_grad()

            # policy loss
            hypo_actions, hypo_logps = self.sac_net(states, deterministic=False)
            q_hypo = torch.min(
                self.sac_net.get_q1_values(states, hypo_actions),
                self.sac_net.get_q2_values(states, hypo_actions)
            )
            policy_loss = (self.alpha * hypo_logps - q_hypo).mean()

            # entropy loss

            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.sac_net.policy_params, 5)
            self.sac_net.policy_optim.step()

            for param in self.sac_net.q_params:
                param.requires_grad = True

            ####################################################################

            self._sac_net_version += 1
            if self._sac_net_version - self._target_sac_net_version == self.update_target_every:
                self._update_target()

    def learn_with_data_parallel(self, batch: dict):
        assert hasattr(self, 'task_queue_client'), "learn_with_data_parallel is invalid before data_parallel is called."

        self._replay_memory.put(
            batch["states"], batch["actions"], batch["rewards"], batch["next_states"], batch["terminals"]
        )
        for _ in range(self.num_epochs):
            worker_id_list = self.task_queue_client.request_workers()
            batch_list = [
                self._replay_memory.sample(self.train_batch_size // len(worker_id_list))
                for i in range(len(worker_id_list))]
            loss_info_by_policy = self.task_queue_client.submit(
                worker_id_list, batch_list, self.get_state(), self._name)
            # build dummy computation graph by `get_batch_loss` before apply gradients.
            # batch_size=2 because torch.nn.functional.batch_norm doesn't support batch_size=1.
            _ = self.get_batch_loss(self._replay_memory.sample(2), explicit_grad=True)
            self.update(loss_info_by_policy[self._name])

    def _update_target(self):
        # soft-update target network
        self.target_sac_net.soft_update(self.sac_net, self.soft_update_coeff)
        self._target_ac_net_version = self._sac_net_version

    def get_state(self):
        return self.sac_net.get_state()

    def set_state(self, state):
        self.sac_net.set_state(state)

    def load(self, path: str):
        """Load the policy state from disk."""
        checkpoint = torch.load(path)
        self.sac_net.set_state(checkpoint["ac_net"])
        self._sac_net_version = checkpoint["ac_net_version"]
        self.target_sac_net.set_state(checkpoint["target_sac_net"])
        self._target_sac_net_version = checkpoint["target_sac_net_version"]
        self._replay_memory = checkpoint["replay_memory"]

    def save(self, path: str):
        """Save the policy state to disk."""
        policy_state = {
            "sac_net": self.sac_net.get_state(),
            "sac_net_version": self._sac_net_version,
            "target_sac_net": self.target_sac_net.get_state(),
            "target_sac_net_version": self._target_sac_net_version,
            "replay_memory": self._replay_memory
        }
        torch.save(policy_state, path)
