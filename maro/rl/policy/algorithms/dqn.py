# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Union

import numpy as np
import torch

from maro.rl.experience import ExperienceManager, PrioritizedSampler
from maro.rl.exploration import DiscreteSpaceExploration, EpsilonGreedyExploration
from maro.rl.model import DiscreteQNet
from maro.rl.policy import AbsCorePolicy


class DQNConfig:
    """Configuration for the DQN algorithm.

    Args:
        reward_discount (float): Reward decay as defined in standard RL terminology.
        target_update_freq (int): Number of training rounds between target model updates.
        train_epochs (int): Number of training epochs per call to ``update()``. Defaults to 1.
        soft_update_coefficient (float): Soft update coefficient, e.g.,
            target_model = (soft_update_coefficient) * eval_model + (1-soft_update_coefficient) * target_model.
            Defaults to 1.0.
        double (bool): If True, the next Q values will be computed according to the double DQN algorithm,
            i.e., q_next = Q_target(s, argmax(Q_eval(s, a))). Otherwise, q_next = max(Q_target(s, a)).
            See https://arxiv.org/pdf/1509.06461.pdf for details. Defaults to False.
    """
    __slots__ = [
        "reward_discount", "target_update_freq", "train_epochs", "gradient_iters", "soft_update_coefficient",
        "double",
    ]

    def __init__(
        self,
        reward_discount: float,
        target_update_freq: int,
        train_epochs: int = 1,
        soft_update_coefficient: float = 0.1,
        double: bool = True
    ):
        self.reward_discount = reward_discount
        self.target_update_freq = target_update_freq
        self.train_epochs = train_epochs
        self.soft_update_coefficient = soft_update_coefficient
        self.double = double


class DQN(AbsCorePolicy):
    """The Deep-Q-Networks algorithm.

    See https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf for details.

    Args:
        q_net (DiscreteQNet): Q-value model.
        experience_manager (ExperienceManager): An experience manager for storing and retrieving experiences
            for training.
        config (DQNConfig): Configuration for DQN algorithm.
        exploration (DiscreteSpaceExploration): Exploration strategy for generating exploratory actions. Defaults to
            an ``EpsilonGreedyExploration`` instance.
    """
    def __init__(
        self,
        q_net: DiscreteQNet,
        experience_manager: ExperienceManager,
        config: DQNConfig,
        exploration: DiscreteSpaceExploration = EpsilonGreedyExploration()
    ):
        if not isinstance(q_net, DiscreteQNet):
            raise TypeError("model must be an instance of 'DiscreteQNet'")

        super().__init__(experience_manager, exploration=exploration)
        self.q_net = q_net
        if self.q_net.trainable:
            self.target_q_net = q_net.copy()
            self.target_q_net.eval()
        else:
            self.target_q_net = None
        self.config = config
        self.device = self.q_net.device
        self._training_counter = 0
        self.prioritized_experience_replay = isinstance(self.experience_manager.sampler, PrioritizedSampler)
        if not self.prioritized_experience_replay:
            self._loss_func = torch.nn.MSELoss()

    def choose_action(self, states) -> Union[int, np.ndarray]:
        with torch.no_grad():
            self.q_net.eval()
            actions, _, num_actions = self.q_net.get_action(states)

        actions = actions.cpu().numpy()
        self.exploration.set_action_space(np.arange(num_actions))
        if self.exploring:
            actions = self.exploration(actions)
        return actions[0] if len(actions) == 1 else actions

    def learn(self):
        assert self.q_net.trainable, "q_net needs to have at least one optimizer registered."
        self.q_net.train()
        for _ in range(self.config.train_epochs):
            # sample from the replay memory
            experience_set = self.experience_manager.get()
            states, next_states = experience_set.states, experience_set.next_states
            actions = torch.from_numpy(np.asarray(experience_set.actions)).to(self.device)
            rewards = torch.from_numpy(np.asarray(experience_set.rewards)).to(self.device)
            if self.prioritized_experience_replay:
                indexes = [info["index"] for info in experience_set.info]
                is_weights = torch.tensor([info["is_weight"] for info in experience_set.info]).to(self.device)

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
                loss = (td_errors * is_weights).mean()
                self.experience_manager.sampler.update(indexes, td_errors.detach().cpu().numpy())
            else:
                loss = self._loss_func(q_values, target_q_values)
            self.q_net.step(loss)

            # soft-update target network
            self._training_counter += 1
            if self._training_counter % self.config.target_update_freq == 0:
                self.target_q_net.soft_update(self.q_net, self.config.soft_update_coefficient)

    def set_state(self, policy_state):
        self.q_net.load_state_dict(policy_state)
        self.target_q_net = self.q_net.copy() if self.q_net.trainable else None
        if self.target_q_net:
            self.target_q_net.eval()

    def get_state(self):
        return self.q_net.state_dict()
