# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable, Union

import numpy as np
import torch

from maro.rl.algorithms import AbsAlgorithm
from maro.rl.experience import ExperienceSet, PrioritizedSampler
from maro.rl.exploration import DiscreteSpaceExploration, EpsilonGreedyExploration
from maro.rl.model import DiscreteQNet


class DQNConfig:
    """Configuration for the DQN algorithm.

    Args:
        reward_discount (float): Reward decay as defined in standard RL terminology.
        update_target_every (int): Number of gradient steps between target model updates.
        soft_update_coeff (float): Soft update coefficient, e.g.,
            target_model = (soft_update_coeff) * eval_model + (1-soft_update_coeff) * target_model.
            Defaults to 1.0.
        double (bool): If True, the next Q values will be computed according to the double DQN algorithm,
            i.e., q_next = Q_target(s, argmax(Q_eval(s, a))). Otherwise, q_next = max(Q_target(s, a)).
            See https://arxiv.org/pdf/1509.06461.pdf for details. Defaults to False.
    """
    __slots__ = ["reward_discount", "update_target_every", "soft_update_coeff", "double"]

    def __init__(
        self,
        reward_discount: float,
        update_target_every: int,
        soft_update_coeff: float = 0.1,
        double: bool = True
    ):
        self.reward_discount = reward_discount
        self.update_target_every = update_target_every
        self.soft_update_coeff = soft_update_coeff
        self.double = double


class DQN(AbsAlgorithm):
    """The Deep-Q-Networks algorithm.

    See https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf for details.

    Args:
        q_net (DiscreteQNet): Q-value model.
        config (DQNConfig): Configuration for DQN algorithm.
        exploration (DiscreteSpaceExploration): Exploration strategy for generating exploratory actions. Defaults to
            an ``EpsilonGreedyExploration`` instance.
        post_step (Callable): Custom function to be called after each gradient step. This can be used for tracking
            the learning progress. The function should have signature (loss, tracker) -> None. Defaults to None.
    """
    def __init__(
        self,
        q_net: DiscreteQNet,
        config: DQNConfig,
        exploration: DiscreteSpaceExploration = EpsilonGreedyExploration(),
        post_step: Callable = None
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
        self.config = config
        self._post_step = post_step
        self.device = self.q_net.device

        self.prioritized_experience_replay = isinstance(self.sampler, PrioritizedSampler)
        if not self.prioritized_experience_replay:
            self._loss_func = torch.nn.MSELoss()

    def choose_action(self, states, explore: bool = True) -> Union[int, np.ndarray]:
        self.q_net.eval()
        with torch.no_grad():
            q_for_all_actions = self.q_net(states)  # (batch_size, num_actions)
            _, actions = q_for_all_actions.max(dim=1)

        actions = actions.cpu().numpy()
        if self.exploration.action_space is None:
            self.exploration.set_action_space(np.arange(q_for_all_actions.shape[1]))
        if explore:
            actions = self.exploration(actions, state=states)
        return actions[0] if len(actions) == 1 else actions

    def get_update_info(self, experience_batch: ExperienceSet):
        return self.q_net.get_gradients(self._get_loss(experience_batch))

    def apply(self, grad_dict: dict):
        self.q_net.apply(grad_dict)

    def _get_loss(self, experience_batch: ExperienceSet):
        states, next_states = experience_batch.states, experience_batch.next_states
        actions = torch.from_numpy(np.asarray(experience_batch.actions)).to(self.device)
        rewards = torch.from_numpy(np.asarray(experience_batch.rewards)).to(self.device)
        if self.prioritized_experience_replay:
            indexes = [info["index"] for info in experience_batch.info]
            is_weights = torch.tensor([info["is_weight"] for info in experience_batch.info]).to(self.device)

        # get target Q values
        with torch.no_grad():
            if self.config.double:
                actions_by_eval_q_net = self.q_net.get_action(next_states)[0]
                next_q_values = self.target_q_net.q_values(next_states, actions_by_eval_q_net)
            else:
                next_q_values = self.target_q_net.get_action(next_states)[1]  # (N,)

        target_q_values = (rewards + self.config.reward_discount * next_q_values).detach()  # (N,)

        # gradient step
        q_values = self.q_net.q_values(states, actions)
        if self.prioritized_experience_replay:
            td_errors = target_q_values - q_values
            self.sampler.update(indexes, td_errors.detach().cpu().numpy())
            return (td_errors * is_weights).mean()
        else:
            return self._loss_func(q_values, target_q_values)

    def learn(self, experience_batch: ExperienceSet):
        assert self.q_net.trainable, "q_net needs to have at least one optimizer registered."
        self.q_net.train()

        loss = self._get_loss(experience_batch)
        self.q_net.step(loss)
        if self._post_step:
            self._post_step(loss.detach().cpu().numpy(), self.tracker)

    def update_target(self, num_steps: int):
        # soft-update target network
        if num_steps % self.config.update_target_every == 0:
            self.target_q_net.soft_update(self.q_net, self.config.soft_update_coeff)

    def set_state(self, policy_state):
        self.q_net.load_state_dict(policy_state["eval"])
        if "target" in policy_state:
            self.target_q_net.load_state_dict(policy_state["target"])

    def get_state(self):
        policy_state = {"eval": self.q_net.state_dict()}
        if self.target_q_net:
            policy_state["target"] = self.target_q_net.state_dict()
        return policy_state
