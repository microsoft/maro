# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from typing import List, Tuple

import numpy as np
import torch

from maro.communication import SessionMessage
from maro.rl.modeling import DiscretePolicyNet
from maro.rl.utils import MsgKey, MsgTag, average_grads, discount_cumsum

from .policy import RLPolicy


class PolicyGradient(RLPolicy):
    class Buffer:
        """Store a sequence of transitions, i.e., a trajectory.

        Args:
            state_dim (int): State vector dimension.
            size (int): Buffer capacity, i.e., the maximum number of stored transitions.
        """
        def __init__(self, state_dim, size: int = 10000):
            self.states = np.zeros((size, state_dim), dtype=np.float32)
            self.values = np.zeros(size, dtype=np.float32)
            self.rewards = np.zeros(size, dtype=np.float32)
            self.terminals = np.zeros(size, dtype=np.bool)
            self.size = size

        def put(self, state: np.ndarray, action: dict, reward: float, terminal: bool = False):
            self.states[self._ptr] = state
            self.values[self._ptr] = action["value"]
            self.rewards[self._ptr] = reward
            self.terminals[self._ptr] = terminal
            # increment pointer
            self._ptr += 1
            if self._ptr == self.size:
                self._ptr = 0

        def get(self):
            terminal = self.terminals[self._ptr - 1]
            traj_slice = slice(self._last_ptr, self._ptr - (not terminal))
            self._last_ptr = self._ptr - (not terminal)
            return {
                "states": self.states[traj_slice],
                "rewards": self.rewards[traj_slice],
                "last_value": self.values[-1]
            }

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
        policy_net: DiscretePolicyNet,
        reward_discount: float,
        grad_iters: int = 1,
        max_trajectory_len: int = 10000,
        get_loss_on_rollout: bool = False,
        device: str = None
    ):
        if not isinstance(policy_net, DiscretePolicyNet):
            raise TypeError("model must be an instance of 'DiscretePolicyNet'")
        super().__init__(name)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.policy_net = policy_net.to(self.device)
        self.reward_discount = reward_discount
        self.grad_iters = grad_iters
        self.max_trajectory_len = max_trajectory_len
        self.get_loss_on_rollout = get_loss_on_rollout

        self._buffer = defaultdict(lambda: self.Buffer(self.policy_net.input_dim, size=self.max_trajectory_len))

    def __call__(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return a list of action information dict given a batch of states.

        An action information dict contains the action itself and the corresponding log-P value.
        """
        self.policy_net.eval()
        with torch.no_grad():
            actions, logps = self.policy_net.get_action(states, greedy=self.greedy)
        actions, logps = actions.cpu().numpy(), logps.cpu().numpy()
        return [{"action": action, "logp": logp} for action, logp in zip(actions, logps)]

    def record(
        self,
        key: str,
        state: np.ndarray,
        action: dict,
        reward: float,
        next_state: np.ndarray,
        terminal: bool
    ):
        self._buffer[key].put(state, action, reward, terminal)

    def get_rollout_info(self):
        """Extract information from the recorded transitions.

        Returns:
            Loss (including gradients) for the latest trajectory segment in the replay buffer if ``get_loss_on_rollout``
            is True or the latest trajectory segment with pre-computed return values.
        """
        if self.get_loss_on_rollout:
            return self.get_batch_loss(self._get_batch(), explicit_grad=True)
        else:
            return self._get_batch()

    def _get_batch(self):
        batch = defaultdict(list)
        for buf in self._buffer:
            trajectory = buf.get()
            rewards = np.append(trajectory["rewards"], trajectory["last_val"])
            batch["states"].append(trajectory["states"])
            # Returns rewards-to-go, to be targets for the value function
            batch["returns"].append(discount_cumsum(rewards, self.reward_discount)[:-1])

        return {key: np.concatenate(vals) for key, vals in batch.items}

    def get_batch_loss(self, batch: dict, explicit_grad: bool = False):
        """Compute AC loss for a data batch.

        Args:
            batch (dict): A batch containing "states" and "returns" as keys.
            explicit_grad (bool): If True, the gradients should be returned as part of the loss information. Defaults
                to False.
        """
        self.policy_net.train()
        returns = torch.from_numpy(np.asarray(batch["returns"])).to(self.device)

        _, logp = self.policy_net(batch["states"])
        loss = -(logp * returns).mean()
        loss_info = {"loss": loss.detach().cpu().numpy() if explicit_grad else loss}
        if explicit_grad:
            loss_info["grad"] = self.policy_net.get_gradients(loss)
        return loss_info

    def update(self, loss_info_list: List[dict]):
        """Update the model parameters with gradients computed by multiple roll-out instances or gradient workers.

        Args:
            loss_info_list (List[dict]): A list of dictionaries containing loss information (including gradients)
                computed by multiple roll-out instances or gradient workers.
        """
        self.policy_net.apply_gradients(average_grads([loss_info["grad"] for loss_info in loss_info_list]))

    def learn(self, batch: dict):
        """Learn from a batch containing data required for policy improvement.

        Args:
            batch (dict): A batch containing "states" and "returns" as keys.
        """
        for _ in range(self.grad_iters):
            self.policy_net.step(self.get_batch_loss(batch)["grad"])

    def improve(self):
        """Learn using data from the buffer."""
        self.learn(self._get_batch())

    def learn_with_data_parallel(self, batch: dict, worker_id_list: list):
        assert hasattr(self, '_proxy'), "learn_with_data_parallel is invalid before data_parallel is called."

        for _ in range(self.grad_iters):
            msg_dict = defaultdict(lambda: defaultdict(dict))
            for i, worker_id in enumerate(worker_id_list):
                sub_batch = {key: batch[key][i::len(worker_id_list)] for key in batch}
                msg_dict[worker_id][MsgKey.GRAD_TASK][self._name] = sub_batch
                msg_dict[worker_id][MsgKey.POLICY_STATE][self._name] = self.get_state()
                # data-parallel
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
            # build dummy computation graph before apply gradients.
            _ = self.get_batch_loss(sub_batch, explicit_grad=True)
            self.policy_net.step(loss_info_by_policy[self._name])

    def set_state(self, policy_state):
        self.policy_net.load_state_dict(policy_state)

    def get_state(self):
        return self.policy_net.state_dict()
