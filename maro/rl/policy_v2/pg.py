# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from typing import List, Tuple

import numpy as np
import torch

from maro.rl.modeling_v2 import DiscretePolicyGradientNetwork
from maro.rl.utils import average_grads, discount_cumsum

from .buffer import Buffer
from .policy_base import SingleRLPolicy
from .policy_interfaces import DiscreteActionMixin


class DiscretePolicyGradient(DiscreteActionMixin, SingleRLPolicy):
    """The vanilla Policy Gradient (VPG) algorithm, a.k.a., REINFORCE.

    Reference: https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch.

    Args:
        name (str): Unique identifier for the policy.
        policy_net (DiscretePolicyNet): Multi-task model that computes action distributions and state values.
            It may or may not have a shared bottom stack.
        reward_discount (float): Reward decay as defined in standard RL terminology.
        grad_iters (int): Number of gradient steps for each batch or set of batches. Defaults to 1.
        max_trajectory_len (int): Maximum trajectory length that can be held by the buffer (for each agent that uses
            this policy). Defaults to 10000.
        get_loss_on_rollout (bool): If True, ``get_rollout_info`` will return the loss information (including gradients)
            for the trajectories stored in the buffers. The loss information, along with that from other roll-out
            instances, can be passed directly to ``update``. Otherwise, it will simply process the trajectories into a
            single data batch that can be passed directly to ``learn``. Defaults to False.
        device (str): Identifier for the torch device. The ``policy net`` will be moved to the specified device. If it
            is None, the device will be set to "cpu" if cuda is unavailable and "cuda" otherwise. Defaults to None.
    """
    def __init__(
        self,
        name: str,
        policy_net: DiscretePolicyGradientNetwork,
        reward_discount: float,
        grad_iters: int = 1,
        max_trajectory_len: int = 10000,
        get_loss_on_rollout: bool = False,
        device: str = None
    ) -> None:
        super(DiscretePolicyGradient, self).__init__(name=name, device=device)

        if not isinstance(policy_net, DiscretePolicyGradientNetwork):
            raise TypeError("model must be an instance of 'DiscretePolicyGradientNetwork'")

        self._policy_net = policy_net.to(self._device)
        self._reward_discount = reward_discount
        self._grad_iters = grad_iters
        self._max_trajectory_len = max_trajectory_len
        self._get_loss_on_rollout = get_loss_on_rollout

        self._buffer = defaultdict(lambda: Buffer(size=self._max_trajectory_len))

    def _call_impl(self, states: np.ndarray) -> List[dict]:
        """Return a list of action information dict given a batch of states.

        An action information dict contains the action itself and the corresponding log-P value.
        """
        actions, logps = self.get_actions_with_logps(states)
        return [{"action": action, "logp": logp} for action, logp in zip(actions, logps)]

    def _call_post_check(self, states: np.ndarray, ret: List[dict]) -> bool:
        return len(ret) == states.shape[0]

    def _get_actions_impl(self, states: np.ndarray) -> np.ndarray:
        return self.get_actions_with_logps(states)[0].reshape(-1, self.action_dim)

    def get_actions_with_logps(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return actions an log-P value based on states.

        Args:
            states (np.ndarray): States with shape [batch_size, state_dim].

        Returns:
            Actions and log-P values, both with shape [batch_size].
        """
        self._policy_net.eval()
        with torch.no_grad():
            states: torch.Tensor = self.ndarray_to_tensor(states)
            actions, logps = self._policy_net.get_actions_and_logps(states, exploring=self._exploring)
        actions, logps = actions.cpu().numpy(), logps.cpu().numpy()
        return actions, logps

    def _get_action_num(self) -> int:
        return self._policy_net.action_num

    def _get_state_dim(self) -> int:
        return self._policy_net.state_dim

    def _get_action_dim(self) -> int:
        return 1

    def record(
        self,
        key: str,
        state: np.ndarray,
        action: dict,
        reward: float,
        next_state: np.ndarray,
        terminal: bool
    ) -> None:
        self._buffer[key].put(state, action, reward, terminal)

    def get_rollout_info(self) -> dict:
        """Extract information from the recorded transitions.

        Returns:
            Loss (including gradients) for the latest trajectory segment in the replay buffer if ``get_loss_on_rollout``
            is True or the latest trajectory segment with pre-computed return values.
        """
        if self._get_loss_on_rollout:
            return self.get_batch_loss(self._get_batch(), explicit_grad=True)
        else:
            return self._get_batch()

    def _get_batch(self) -> dict:
        batch = defaultdict(list)
        for buf in self._buffer:
            trajectory = buf.get()
            rewards = np.append(trajectory["rewards"], trajectory["last_val"])
            batch["states"].append(trajectory["states"])
            # Returns rewards-to-go, to be targets for the value function
            batch["returns"].append(discount_cumsum(rewards, self._reward_discount)[:-1])

        return {key: np.concatenate(vals) for key, vals in batch.items()}

    def get_batch_loss(self, batch: dict, explicit_grad: bool = False) -> dict:
        """Compute AC loss for a data batch.

        Args:
            batch (dict): A batch containing "states" and "returns" as keys.
            explicit_grad (bool): If True, the gradients should be returned as part of the loss information. Defaults
                to False.
        """
        self._policy_net.train()
        returns = self.ndarray_to_tensor(np.asarray(batch["returns"]))

        logps = self._policy_net.get_logps(batch["states"])
        loss = -(logps * returns).mean()
        loss_info = {"loss": loss.detach().cpu().numpy() if explicit_grad else loss}
        if explicit_grad:
            loss_info["grad"] = self._policy_net.get_gradients(loss)
        return loss_info

    def update(self, loss_info_list: List[dict]) -> None:
        """Update the model parameters with gradients computed by multiple roll-out instances or gradient workers.

        Args:
            loss_info_list (List[dict]): A list of dictionaries containing loss information (including gradients)
                computed by multiple roll-out instances or gradient workers.
        """
        self._policy_net.apply_gradients(average_grads([loss_info["grad"] for loss_info in loss_info_list]))

    def learn(self, batch: dict) -> None:
        """Learn from a batch containing data required for policy improvement.

        Args:
            batch (dict): A batch containing "states" and "returns" as keys.
        """
        for _ in range(self._grad_iters):
            self._policy_net.step(self.get_batch_loss(batch)["grad"])

    def improve(self) -> None:
        """Learn using data from the buffer."""
        self.learn(self._get_batch())

    def learn_with_data_parallel(self, batch: dict) -> None:
        assert self._task_queue_client, "learn_with_data_parallel is invalid before data_parallel is called."

        for _ in range(self._grad_iters):
            worker_id_list = self._task_queue_client.request_workers()
            batch_list = [
                {key: batch[key][i::len(worker_id_list)] for key in batch} for i in range(len(worker_id_list))]
            loss_info_by_policy = self._task_queue_client.submit(
                worker_id_list, batch_list, self.get_state(), self._name)
            # build dummy computation graph before apply gradients.
            _ = self.get_batch_loss(batch_list[0], explicit_grad=True)
            self._policy_net.step(loss_info_by_policy[self._name])

    def get_state(self) -> object:
        return self._policy_net.get_state()

    def set_state(self, policy_state: object) -> None:
        self._policy_net.set_state(policy_state)

    def load(self, path: str) -> None:
        """Load the policy state from disk."""
        self._policy_net.set_state(torch.load(path))

    def save(self, path: str) -> None:
        """Save the policy state to disk."""
        torch.save(self._policy_net.get_state(), path)
