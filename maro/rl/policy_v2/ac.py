# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from collections import defaultdict
from typing import Callable, List, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from maro.communication import SessionMessage
from maro.rl.modeling_v2 import DiscreteVActorCriticNet
from maro.rl.utils import MsgKey, MsgTag, average_grads, discount_cumsum

from .buffer import Buffer
from .policy_base import RLPolicyV2
from .policy_interfaces import DiscreteActionMixin, VNetworkMixin


class DiscreteActorCritic(VNetworkMixin, DiscreteActionMixin, RLPolicyV2):
    """
    Actor Critic algorithm with separate policy and value models.

    References:
        https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch.
        https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f

    Args:
        name (str): Unique identifier for the policy.
        ac_net (DiscreteACNet): Multi-task model that computes action distributions and state values.
        reward_discount (float): Reward decay as defined in standard RL terminology.
        grad_iters (int): Number of gradient steps for each batch or set of batches. Defaults to 1.
        critic_loss_cls: A string indicating a loss class provided by torch.nn or a custom loss class for computing
            the critic loss. If it is a string, it must be a key in ``TORCH_LOSS``. Defaults to "mse".
        min_logp (float): Lower bound for clamping logP values during learning. This is to prevent logP from becoming
            very large in magnitude and causing stability issues. Defaults to None, which means no lower bound.
        critic_loss_coef (float): Coefficient for critic loss in total loss. Defaults to 1.0.
        entropy_coef (float): Coefficient for the entropy term in total loss. Defaults to None, in which case the
            total loss will not include an entropy term.
        clip_ratio (float): Clip ratio in the PPO algorithm (https://arxiv.org/pdf/1707.06347.pdf). Defaults to None,
            in which case the actor loss is calculated using the usual policy gradient theorem.
        lam (float): Lambda value for generalized advantage estimation (TD-Lambda). Defaults to 0.9.
        max_trajectory_len (int): Maximum trajectory length that can be held by the buffer (for each agent that uses
            this policy). Defaults to 10000.
        get_loss_on_rollout (bool): If True, ``get_rollout_info`` will return the loss information (including gradients)
            for the trajectories stored in the buffers. The loss information, along with that from other roll-out
            instances, can be passed directly to ``update``. Otherwise, it will simply process the trajectories into a
            single data batch that can be passed directly to ``learn``. Defaults to False.
        device (str): Identifier for the torch device. The ``ac_net`` will be moved to the specified device. If it is
            None, the device will be set to "cpu" if cuda is unavailable and "cuda" otherwise. Defaults to None.
    """
    def __init__(
        self,
        name: str,
        ac_net: DiscreteVActorCriticNet,
        reward_discount: float,
        grad_iters: int = 1,
        critic_loss_cls: Callable = None,
        min_logp: float = None,
        critic_loss_coef: float = 1.0,
        entropy_coef: float = .0,
        clip_ratio: float = None,
        lam: float = 0.9,
        max_trajectory_len: int = 10000,
        get_loss_on_rollout: bool = False,
        device: str = None
    ) -> None:
        if not isinstance(ac_net, DiscreteVActorCriticNet):
            raise TypeError("model must be an instance of 'DiscreteVActorCriticNet'")

        super(DiscreteActorCritic, self).__init__(name=name, device=device)

        self._ac_net = ac_net.to(self._device)
        self._reward_discount = reward_discount
        self._grad_iters = grad_iters
        self._critic_loss_func = critic_loss_cls() if critic_loss_cls is not None else torch.nn.MSELoss()
        self._min_logp = min_logp
        self._critic_loss_coef = critic_loss_coef
        self._entropy_coef = entropy_coef
        self._clip_ratio = clip_ratio
        self._lam = lam
        self._max_trajectory_len = max_trajectory_len
        self._get_loss_on_rollout = get_loss_on_rollout

        self._buffer = defaultdict(lambda: Buffer(size=self._max_trajectory_len))

    def _call_impl(self, states: np.ndarray) -> List[dict]:
        """Return a list of action information dict given a batch of states.

        An action information dict contains the action itself, the corresponding log-P value and the corresponding
        state value.
        """
        actions, logps, values = self.get_actions_with_logps_and_values(states)
        return [
            {"action": action, "logp": logp, "value": value} for action, logp, value in zip(actions, logps, values)
        ]

    def get_actions_with_logps_and_values(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._ac_net.eval()
        states = torch.from_numpy(states).to(self._device)
        if len(states.shape) == 1:
            states = states.unsqueeze(dim=0)
        with torch.no_grad():
            actions, logps = self._ac_net.get_actions_and_logps(states, exploring=self._exploring)
            values = self._get_v_critic(states)
        actions, logps, values = actions.cpu().numpy(), logps.cpu().numpy(), values.cpu().numpy()
        return actions, logps, values

    def _get_v_critic(self, states: torch.Tensor) -> torch.Tensor:
        return self._ac_net.v_critic(states)

    def _get_v_values(self, states: np.ndarray) -> np.ndarray:
        return self._get_v_critic(torch.Tensor(states)).numpy()

    def learn_with_data_parallel(self, batch: dict, worker_id_list: list) -> None:
        assert hasattr(self, '_proxy'), "learn_with_data_parallel is invalid before data_parallel is called."
        for _ in range(self._grad_iters):
            msg_dict = defaultdict(lambda: defaultdict(dict))
            sub_batch = {}
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
                            loss_info_by_policy[policy_name].append(loss_info)
                        else:
                            raise TypeError(f"Wrong type of loss_info: {type(loss_info)}")
                    dones += 1
                    if dones == len(msg_dict):
                        break
            # build dummy computation graph by `get_batch_loss` before apply gradients.
            _ = self.get_batch_loss(sub_batch, explicit_grad=True)
            self.update(loss_info_by_policy[self._name])

    def _get_action_num(self) -> int:
        return self._ac_net.action_num

    def _get_state_dim(self) -> int:
        return self._ac_net.state_dim

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
        if self._get_loss_on_rollout:
            return self.get_batch_loss(self._get_batch(), explicit_grad=True)
        else:
            return self._get_batch()

    def _get_batch(self) -> dict:
        batch = defaultdict(list)
        for buf in self._buffer.values():
            trajectory = buf.get()
            values = np.append(trajectory["values"], trajectory["last_value"])
            rewards = np.append(trajectory["rewards"], trajectory["last_value"])
            deltas = rewards[:-1] + self._reward_discount * values[1:] - values[:-1]
            batch["states"].append(trajectory["states"])
            batch["actions"].append(trajectory["actions"])
            # Returns rewards-to-go, to be targets for the value function
            batch["returns"].append(discount_cumsum(rewards, self._reward_discount)[:-1])
            # Generalized advantage estimation using TD(Lambda)
            batch["advantages"].append(discount_cumsum(deltas, self._reward_discount * self._lam))
            batch["logps"].append(trajectory["logps"])

        return {key: np.concatenate(vals) for key, vals in batch.items()}

    def get_batch_loss(self, batch: dict, explicit_grad: bool = False) -> dict:
        self._ac_net.train()
        states = torch.from_numpy(batch["states"]).to(self._device)
        actions = torch.from_numpy(batch["actions"]).to(self._device).long()
        logp_old = torch.from_numpy(batch["logps"]).to(self._device)
        returns = torch.from_numpy(batch["returns"]).to(self._device)
        advantages = torch.from_numpy(batch["advantages"]).to(self._device)

        action_probs = self._ac_net.get_probs(states)
        state_values = self._get_v_critic(states)
        state_values = state_values.squeeze()

        # actor loss
        logp = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())  # (N,)
        logp = torch.clamp(logp, min=self._min_logp, max=.0)
        if self._clip_ratio is not None:
            ratio = torch.exp(logp - logp_old)
            clipped_ratio = torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio)
            actor_loss = -(torch.min(ratio * advantages, clipped_ratio * advantages)).mean()
        else:
            actor_loss = -(logp * advantages).mean()

        # critic_loss
        critic_loss = self._critic_loss_func(state_values, returns)
        # entropy
        entropy = -Categorical(action_probs).entropy().mean() if self._entropy_coef else 0

        # total loss
        loss = actor_loss + self._critic_loss_coef * critic_loss + self._entropy_coef * entropy

        loss_info = {
            "actor_loss": actor_loss.detach().cpu().numpy(),
            "critic_loss": critic_loss.detach().cpu().numpy(),
            "entropy": entropy.detach().cpu().numpy() if self._entropy_coef else .0,
            "loss": loss.detach().cpu().numpy() if explicit_grad else loss
        }
        if explicit_grad:
            loss_info["grad"] = self._ac_net.get_gradients(loss)

        return loss_info

    def data_parallel(self, *args, **kwargs) -> None:
        pass  # TODO

    def update(self, loss_info_list: List[dict]) -> None:
        self._ac_net.apply_gradients(average_grads([loss_info["grad"] for loss_info in loss_info_list]))

    def learn(self, batch: dict) -> None:
        for _ in range(self._grad_iters):
            self._ac_net.step(self.get_batch_loss(batch)["loss"])

    def improve(self) -> None:
        self.learn(self._get_batch())

    def get_state(self) -> object:
        return self._ac_net.get_state()

    def set_state(self, policy_state: object) -> None:
        self._ac_net.set_state(policy_state)

    def load(self, path: str) -> None:
        self._ac_net.set_state(torch.load(path))

    def save(self, path: str) -> None:
        torch.save(self._ac_net.get_state(), path)
