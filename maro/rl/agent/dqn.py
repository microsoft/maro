# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Union

import numpy as np
import torch

from maro.rl.model import SimpleMultiHeadModel
from maro.rl.storage import SimpleStore
from maro.rl.utils import get_max, get_sampler_cls, get_td_errors, get_torch_loss_cls, select_by_actions
from maro.utils.exception.rl_toolkit_exception import UnrecognizedTask

from .abs_agent import AbsAgent


class DQNConfig:
    """Configuration for the DQN algorithm.

    Args:
        reward_discount (float): Reward decay as defined in standard RL terminology.
        target_update_freq (int): Number of training rounds between target model updates. 
        train_iters (int): Number of batches to train the model on in each call to ``learn``.
        batch_size (int): Experience minibatch size.
        sampler_cls: A string indicating the sampler class or a custom sampler class that provides the ``sample`` interface.
            Defaults to "uniform".
        sampler_params (dict): Parameters for the sampler class. Defaults to None.
        epsilon (float): Exploration rate for epsilon-greedy exploration. Defaults to None.
        soft_update_coefficient (float): Soft update coefficient, e.g., 
            target_model = (soft_update_coefficient) * eval_model + (1-soft_update_coefficient) * target_model.
            Defaults to 1.0.
        double (bool): If True, the next Q values will be computed according to the double DQN algorithm,
            i.e., q_next = Q_target(s, argmax(Q_eval(s, a))). Otherwise, q_next = max(Q_target(s, a)).
            See https://arxiv.org/pdf/1509.06461.pdf for details. Defaults to False.
        advantage_type (str): Advantage mode for the dueling architecture. Defaults to None, in which
            case it is assumed that the regular Q-value model is used.
        loss_cls: A string indicating a loss class provided by torch.nn or a custom loss class. If it is a string,
            it must be a key in ``TORCH_LOSS``. Defaults to "mse".
    """
    __slots__ = [
        "reward_discount", "target_update_freq", "train_iters", "batch_size", "sampler_cls", "sampler_params",
        "epsilon", "soft_update_coefficient", "double", "advantage_type", "loss_func"
    ]

    def __init__(
        self,
        reward_discount: float,
        target_update_freq: int,
        train_iters: int,
        batch_size: int,
        sampler_cls="uniform",
        sampler_params=None,
        epsilon: float = .0,
        soft_update_coefficient: float = 0.1,
        double: bool = True,
        advantage_type: str = None,
        loss_cls="mse"
    ):
        self.reward_discount = reward_discount
        self.target_update_freq = target_update_freq
        self.train_iters = train_iters
        self.batch_size = batch_size
        self.sampler_cls = get_sampler_cls(sampler_cls)
        self.sampler_params = sampler_params if sampler_params else {}
        self.epsilon = epsilon
        self.soft_update_coefficient = soft_update_coefficient
        self.double = double
        self.advantage_type = advantage_type
        self.loss_func = get_torch_loss_cls(loss_cls)()


class DQN(AbsAgent):
    """The Deep-Q-Networks algorithm.

    See https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf for details.

    Args:
        model (SimpleMultiHeadModel): Q-value model.
        config (DQNConfig): Configuration for DQN algorithm.
        experience_memory_size (int): Size of the experience memory. If it is -1, the experience memory is of
            unlimited size.
        experience_memory_overwrite_type (str): A string indicating how experiences in the experience memory are
            to be overwritten after its capacity has been reached. Must be "rolling" or "random".
        flush_experience_memory_after_step (bool): If True, the experience memory will be flushed after each call
            to ``step``. Defaults to False.
        min_new_experiences_to_trigger_learning (int): Minimum number of new experiences required to trigger learning.
            Defaults to 1.
        min_experience_memory_size (int): Minimum number of experiences in the experience memory required for training.
            Defaults to 1.
    """
    def __init__(
        self,
        model: SimpleMultiHeadModel,
        config: DQNConfig,
        experience_memory_size: int,
        experience_memory_overwrite_type: str,
        flush_experience_memory_after_step: bool = False,
        min_new_experiences_to_trigger_learning: int = 1,
        min_experience_memory_size: int = 1
    ):
        if (config.advantage_type is not None and
                (model.task_names is None or set(model.task_names) != {"state_value", "advantage"})):
            raise UnrecognizedTask(
                f"Expected model task names 'state_value' and 'advantage' since dueling DQN is used, "
                f"got {model.task_names}"
            )
        super().__init__(
            model, config, experience_memory_size, experience_memory_overwrite_type,
            flush_experience_memory_after_step,
            min_new_experiences_to_trigger_learning=min_new_experiences_to_trigger_learning,
            min_experience_memory_size=min_experience_memory_size    
        )
        self._sampler = self.config.sampler_cls(self.experience_memory, **self.config.sampler_params)
        self._training_counter = 0
        self._target_model = model.copy() if model.trainable else None

    def choose_action(self, state: np.ndarray) -> Union[int, np.ndarray]:
        state = torch.from_numpy(state)
        if self.device:
            state = state.to(self.device)
        is_single = len(state.shape) == 1
        if is_single:
            state = state.unsqueeze(dim=0)

        q_values = self._get_q_values(state, training=False)
        num_actions = q_values.shape[1]
        greedy_action = q_values.argmax(dim=1).data.cpu()
        # No exploration
        if self.config.epsilon == .0:
            return greedy_action.item() if is_single else greedy_action.numpy()

        if is_single:
            return greedy_action if np.random.random() > self.config.epsilon else np.random.choice(num_actions)

        # batch inference
        return np.array([
            act if np.random.random() > self.config.epsilon else np.random.choice(num_actions)
            for act in greedy_action
        ])

    def step(self):
        print(f"experience memory size: {len(self.experience_memory)}")
        for _ in range(self.config.train_iters):
            # sample from the replay memory
            indexes, batch = self._sampler.sample(self.config.batch_size)
            states = torch.from_numpy(np.asarray(batch["S"])).to(self.device)
            actions = torch.from_numpy(np.asarray(batch["A"])).to(self.device)
            rewards = torch.from_numpy(np.asarray(batch["R"])).to(self.device)
            next_states = torch.from_numpy(np.asarray(batch["S_"])).to(self.device)

            q_all = self._get_q_values(states)
            q = select_by_actions(q_all, actions)
            next_q_all_target = self._get_q_values(next_states, is_eval=False, training=False)
            if self.config.double:
                next_q_all_eval = self._get_q_values(next_states, training=False)
                next_q = select_by_actions(next_q_all_target, next_q_all_eval.max(dim=1)[1])  # (N,)
            else:
                next_q, _ = get_max(next_q_all_target)  # (N,)

            loss = get_td_errors(q, next_q, rewards, self.config.reward_discount, loss_func=self.config.loss_func)
            self.model.step(loss.mean())
            self._training_counter += 1
            if self._training_counter % self.config.target_update_freq == 0:
                self._target_model.soft_update(self.model, self.config.soft_update_coefficient)

            # update auxillary info for the next round of sampling
            self._sampler.update(indexes, loss.detach().numpy())

    def set_exploration_params(self, epsilon):
        self.config.epsilon = epsilon

    def _get_q_values(self, states: torch.Tensor, is_eval: bool = True, training: bool = True):
        output = self.model(states, training=training) if is_eval else self._target_model(states, training=False)
        if self.config.advantage_type is None:
            return output
        else:
            state_values = output["state_value"]
            advantages = output["advantage"]
            # Use mean or max correction to address the identifiability issue
            corrections = advantages.mean(1) if self.config.advantage_type == "mean" else advantages.max(1)[0]
            return state_values + advantages - corrections.unsqueeze(1)
