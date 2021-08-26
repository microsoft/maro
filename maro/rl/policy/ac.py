# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

import numpy as np
import torch
from torch.distributions import Categorical

from maro.rl.types import DiscreteACNet
from maro.rl.utils import discount_cumsum, get_torch_loss_cls

from .policy import RLPolicy


class ActorCritic(RLPolicy):
    """Actor Critic algorithm with separate policy and value models.

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
        critic_loss_coeff (float): Coefficient for critic loss in total loss. Defaults to 1.0.
        entropy_coeff (float): Coefficient for the entropy term in total loss. Defaults to None, in which case the
            total loss will not include an entropy term.
        clip_ratio (float): Clip ratio in the PPO algorithm (https://arxiv.org/pdf/1707.06347.pdf). Defaults to None,
            in which case the actor loss is calculated using the usual policy gradient theorem.
    """

    class Buffer(RLPolicy.Buffer):
        """Sequence of transitions for an agent.

        Args:
            states: Sequence of ``State`` objects traversed during simulation.
            actions: Sequence of actions taken in response to the states.
            rewards: Sequence of rewards received as a result of the actions.
            info: Sequence of each transition's auxillary information.
        """
        def __init__(self, state_dim, action_dim: int = 1, max_len: int = 10000):
            super.__init__(state_dim, action_dim=1, max_len=max_len)
            self.logps = np.zeros(max_len, dtype=np.float32)
            self.values = np.zeros(max_len, dtype=np.float32)

        def store(self, state: np.ndarray, action: dict, reward: float, terminal: bool = False):
            self.states[self._ptr] = state
            self.actions[self._ptr] = action["action"]
            self.logps[self._ptr] = action["logp"]
            self.values[self._ptr] = action["value"]
            self.rewards[self._ptr] = reward
            self.terminal[self._ptr] = terminal
            # increment pointer
            self._ptr += 1
            if self._ptr == self.max_len:
                self._ptr = 0

        def get(self):
            traj_slice = slice(self._last_ptr, self._ptr)
            self._last_ptr = self._ptr
            return {
                "states": self.states[traj_slice],
                "actions": self.actions[traj_slice],
                "logps": self.logps[traj_slice],
                "values": self.values[traj_slice],
                "rewards": self.rewards[traj_slice],
                "terminal": self.terminal[self._ptr - 1]
            }


    class Batch(RLPolicy.Batch):

        __slots__ = ["states", "actions", "returns", "advantages", "logps"]

        def __init__(
            self,
            states: np.ndarray,
            actions: np.ndarray,
            returns: np.ndarray,
            advantages: np.ndarray,
            logps: np.ndarray
        ):
            super().__init__()
            self.states = states
            self.actions = actions
            self.returns = returns
            self.advantages = advantages
            self.logps = logps

        @property
        def size(self):
            return len(self.states)


    class LossInfo(RLPolicy.LossInfo):

        __slots__ = ["actor_loss", "critic_loss", "entropy"]

        def __init__(self, loss, actor_loss, critic_loss, entropy, grad=None):
            super().__init__(loss, grad)
            self.loss = loss
            self.actor_loss = actor_loss
            self.critic_loss = critic_loss
            self.entropy = entropy
            self.grad = grad    


    def __init__(
        self,
        name: str,
        ac_net: DiscreteACNet,
        reward_discount: float,
        grad_iters: int = 1,
        critic_loss_cls="mse",
        min_logp: float = None,
        critic_loss_coeff: float = 1.0,
        entropy_coeff: float = None,
        clip_ratio: float = None,
        lam: float = 0.9,
        get_loss_on_rollout_finish: bool = False,
        remote: bool = False
    ):
        if not isinstance(ac_net, DiscreteACNet):
            raise TypeError("model must be an instance of 'DiscreteACNet'")

        super().__init__(name, remote=remote)
        self.ac_net = ac_net
        self.device = self.ac_net.device
        self.reward_discount = reward_discount
        self.grad_iters = grad_iters
        self.critic_loss_func = get_torch_loss_cls(critic_loss_cls)()
        self.min_logp = min_logp
        self.critic_loss_coeff = critic_loss_coeff
        self.entropy_coeff = entropy_coeff
        self.clip_ratio = clip_ratio
        self.lam = lam
        self._get_loss_on_rollout_finish = get_loss_on_rollout_finish

    def choose_action(self, states):
        """Return actions and log probabilities for given states."""
        self.ac_net.eval()
        with torch.no_grad():
            actions, logps, values = self.ac_net.get_action(states)
        actions, logps, values = actions.cpu().numpy(), logps.cpu().numpy(), values.cpu().numpy()
        if len(actions) == 1:
            return {"action": actions[0], "logp": logps[0], "value": values[0]}
        else:
            return [
                {"action": action, "logp": logp, "value": value} for action, logp, value in zip(actions, logps, values)
            ]

    def get_rollout_info(self, trajectory: dict):
        if self._get_loss_on_rollout_finish:
            return self.get_batch_loss(self._preprocess(trajectory), explicit_grad=True)
        else:
            return trajectory

    def _preprocess(self, trajectory: dict):
        if trajectory["terminal"]:
            states = trajectory["states"]
            actions = trajectory["actions"]
            logps = trajectory["logps"]
            values = np.append(trajectory["values"], .0)
            rewards = np.append(trajectory["rewards"], .0)
        else:
            states = trajectory["states"][:-1]
            actions = trajectory["actions"][:-1]
            logps = trajectory["logps"][:-1]
            values = trajectory["values"]
            rewards = np.append(trajectory["rewards"][:-1], trajectory["values"][-1])

        # Generalized advantage estimation using TD(Lambda)
        deltas = rewards[:-1] + self.reward_discount * values[1:] - values[:-1]
        advantages = discount_cumsum(deltas, self.reward_discount * self.lam)
        # Returns rewards-to-go, to be targets for the value function
        returns = discount_cumsum(rewards, self.reward_discount)[:-1]
        return self.Batch(states, actions, returns, advantages, logps)

    def get_batch_loss(self, batch: Batch, explicit_grad: bool = False):
        assert self.ac_net.trainable, "ac_net needs to have at least one optimizer registered."
        self.ac_net.train()
        actions = torch.from_numpy(batch.actions).to(self.device)
        logp_old = torch.from_numpy(batch.logps).to(self.device)
        returns = torch.from_numpy(batch.returns).to(self.device)
        advantages = torch.from_numpy(batch.advantages).to(self.device)

        action_probs, state_values = self.ac_net(batch.states)
        state_values = state_values.squeeze()

        # actor loss
        logp = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())  # (N,)
        logp = torch.clamp(logp, min=self.min_logp, max=.0)
        if self.clip_ratio is not None:
            ratio = torch.exp(logp - logp_old)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            actor_loss = -(torch.min(ratio * advantages, clipped_ratio * advantages)).mean()
        else:
            actor_loss = -(logp * advantages).mean()

        # critic_loss
        critic_loss = self.critic_loss_func(state_values, returns)

        # entropy
        if self.entropy_coeff is not None:
            entropy = -Categorical(action_probs).entropy().mean()
        else:
            entropy = 0

        # total loss
        loss = actor_loss + self.critic_loss_coeff * critic_loss + self.entropy_coeff * entropy
        grad = self.ac_net.get_gradients(loss) if explicit_grad else None
        return self.LossInfo(actor_loss, critic_loss, entropy, loss, grad=grad)

    def update_with_multi_loss_info(self, loss_info_list: List[LossInfo]):
        """Apply gradients to the underlying parameterized model."""
        self.ac_net.apply_gradients([loss_info.grad for loss_info in loss_info_list])

    def learn_from_multi_trajectories(self, trajectories: List[dict]):
        if self.remote:
            # TODO: distributed grad computation
            pass
        else:
            batches = [self._preprocess(traj) for traj in trajectories]
            for _ in range(self.grad_iters):
                self.update_with_multi_loss_info([self.get_batch_loss(batch, explicit_grad=True) for batch in batches])

    def set_state(self, policy_state):
        self.ac_net.load_state_dict(policy_state)

    def get_state(self):
        return self.ac_net.state_dict()
