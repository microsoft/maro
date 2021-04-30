# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import namedtuple
from typing import Union

import numpy as np
import torch

from maro.rl.experience import ExperienceMemory, ExperienceSet
from maro.rl.model import QNetForDiscreteActionSpace
from maro.rl.policy import AbsCorePolicy, TrainingLoopConfig
from maro.rl.utils import get_torch_loss_cls


class DQNConfig:
    """Configuration for the DQN algorithm.

    Args:
        reward_discount (float): Reward decay as defined in standard RL terminology.
        target_update_freq (int): Number of training rounds between target model updates.
        soft_update_coefficient (float): Soft update coefficient, e.g.,
            target_model = (soft_update_coefficient) * eval_model + (1-soft_update_coefficient) * target_model.
            Defaults to 1.0.
        double (bool): If True, the next Q values will be computed according to the double DQN algorithm,
            i.e., q_next = Q_target(s, argmax(Q_eval(s, a))). Otherwise, q_next = max(Q_target(s, a)).
            See https://arxiv.org/pdf/1509.06461.pdf for details. Defaults to False.
        loss_cls: A string indicating a loss class provided by torch.nn or a custom loss class. If it is a string,
            it must be a key in ``TORCH_LOSS``. Defaults to "mse".
    """
    __slots__ = [
        "reward_discount", "target_update_freq", "train_iters", "batch_size", "sampler_cls", "sampler_params",
        "soft_update_coefficient", "double", "loss_func"
    ]

    def __init__(
        self,
        reward_discount: float,
        target_update_freq: int,
        soft_update_coefficient: float = 0.1,
        double: bool = True,
        loss_cls="mse"
    ):
        self.reward_discount = reward_discount
        self.target_update_freq = target_update_freq
        self.soft_update_coefficient = soft_update_coefficient
        self.double = double
        self.loss_func = get_torch_loss_cls(loss_cls)()


class DQN(AbsCorePolicy):
    """The Deep-Q-Networks algorithm.

    See https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf for details.

    Args:
        model (QNetForDiscreteActionSpace): Q-value model.
        config (DQNConfig): Configuration for DQN algorithm.
    """
    def __init__(
        self,
        q_net: QNetForDiscreteActionSpace,
        experience_memory: ExperienceMemory,
        generic_config: TrainingLoopConfig,
        special_config: DQNConfig,
    ):
        if not isinstance(q_net, QNetForDiscreteActionSpace):
            raise TypeError("model must be an instance of 'QNetForDiscreteActionSpace'")

        super().__init__(experience_memory, generic_config, special_config)
        self.q_net = q_net
        self.target_q_net = q_net.copy() if q_net.trainable else None
        self.target_q_net.eval()
        self._training_counter = 0

    def choose_action(self, states) -> Union[int, np.ndarray]:
        with torch.no_grad():
            self.q_net.eval()
            actions, _ = self.q_net.choose_action(states)

        actions = actions.cpu().numpy()
        return actions[0] if len(actions) == 1 else actions

    def learn(self, experience_set: ExperienceSet):
        if not isinstance(experience_set, ExperienceSet):
            raise TypeError(
                f"Expected experience object of type AbsCorePolicy.experience_type, got {type(experience_set)}"
            )

        self.q_net.train()   
        # sample from the replay memory
        states, next_states = experience_set.states, experience_set.next_states
        actions = torch.from_numpy(np.asarray(experience_set.actions))
        rewards = torch.from_numpy(np.asarray(experience_set.rewards))
        q_values = self.q_net.q_values(states, actions)
        # get next Q values
        with torch.no_grad():
            if self.special_config.double:
                next_q_values = self.target_q_net.q_values(next_states, self.q_net.choose_action(next_states)[0])
            else:
                next_q_values = self.target_q_net.choose_action(next_states)[1]  # (N,)

        # get TD errors
        target_q_values = (rewards + self.special_config.reward_discount * next_q_values).detach()  # (N,)
        loss = self.special_config.loss_func(q_values, target_q_values)

        # train and update target if necessary 
        self.q_net.step(loss.mean())
        self._training_counter += 1
        if self._training_counter % self.special_config.target_update_freq == 0:
            self.target_q_net.soft_update(self.q_net, self.special_config.soft_update_coefficient)

    def load_state(self, policy_state):
        self.q_net.load_state_dict(policy_state)
        self.target_q_net = self.q_net.copy() if self.q_net.trainable else None
        self.target_q_net.eval()

    def state(self):
        return self.q_net.state_dict()
