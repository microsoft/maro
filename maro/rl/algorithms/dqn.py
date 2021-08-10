# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import namedtuple
from typing import Union

import numpy as np
import torch

from maro.rl.algorithms import AbsAlgorithm
from maro.rl.experience import ExperienceBatch
from maro.rl.exploration import DiscreteSpaceExploration, EpsilonGreedyExploration
from maro.rl.model import DiscreteQNet

DQNLossInfo = namedtuple("DQNLossInfo", ["td_error", "loss", "grad"])


class DQN(AbsAlgorithm):
    """The Deep-Q-Networks algorithm.

    See https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf for details.

    Args:
        q_net (DiscreteQNet): Q-value model.
        reward_discount (float): Reward decay as defined in standard RL terminology.
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
        reward_discount: float,
        update_target_every: int,
        soft_update_coeff: float = 0.1,
        double: bool = True,
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
        self.update_target_every = update_target_every
        self.soft_update_coeff = soft_update_coeff
        self.double = double
        self.device = self.q_net.device
        self._loss_func = torch.nn.MSELoss()
        self._experience_memory = None

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

    def learn(self, batch: ExperienceBatch, inplace: bool = True):
        assert self.q_net.trainable, "q_net needs to have at least one optimizer registered."
        # If data is an ExperienceSet, get DQN loss from the batch and backprop it throught the network.
        self.q_net.train()
        states, next_states = batch.data.states, batch.data.next_states
        actions = torch.from_numpy(np.asarray(batch.data.actions)).to(self.device)
        rewards = torch.from_numpy(np.asarray(batch.data.rewards)).to(self.device)

        # get target Q values
        with torch.no_grad():
            if self.double:
                actions_by_eval_q_net = self.q_net.get_action(next_states)[0]
                next_q_values = self.target_q_net.q_values(next_states, actions_by_eval_q_net)
            else:
                next_q_values = self.target_q_net.get_action(next_states)[1]  # (N,)

        target_q_values = (rewards + self.reward_discount * next_q_values).detach()  # (N,)

        # loss computation
        q_values = self.q_net.q_values(states, actions)
        td_errors = target_q_values - q_values
        loss = self._loss_func(q_values, target_q_values)

        if inplace:
            self.q_net.step(loss)
            grad = None
            self._q_net_version += 1
            if self._q_net_version - self._target_q_net_version == self.update_target_every:
                self._update_target()
        else:
            grad = self.q_net.get_gradients(loss)

        return DQNLossInfo(td_errors, loss, grad), batch.indexes

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
