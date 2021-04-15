# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable, Tuple

import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import MSELoss

from maro.rl.model import SimpleMultiHeadModel
from maro.rl.utils import get_log_prob, get_torch_loss_cls
from maro.utils.exception.rl_toolkit_exception import UnrecognizedTask

from .abs_agent import AbsAgent


class ActorCriticConfig:
    """Configuration for the Actor-Critic algorithm.

    Args:
        reward_discount (float): Reward decay as defined in standard RL terminology.
        critic_loss_cls: A string indicating a loss class provided by torch.nn or a custom loss class for computing
            the critic loss. If it is a string, it must be a key in ``TORCH_LOSS``. Defaults to "mse".
        train_iters (int): Number of gradient descent steps per call to ``train``.
        actor_loss_coefficient (float): The coefficient for actor loss in the total loss function, e.g.,
            loss = critic_loss + ``actor_loss_coefficient`` * actor_loss. Defaults to 1.0.
        clip_ratio (float): Clip ratio in the PPO algorithm (https://arxiv.org/pdf/1707.06347.pdf). Defaults to None,
            in which case the actor loss is calculated using the usual policy gradient theorem.
    """
    __slots__ = ["reward_discount", "critic_loss_func", "train_iters", "actor_loss_coefficient", "clip_ratio"]

    def __init__(
        self,
        reward_discount: float,
        train_iters: int,
        critic_loss_cls="mse",
        actor_loss_coefficient: float = 1.0,
        clip_ratio: float = None
    ):
        self.reward_discount = reward_discount
        self.critic_loss_func = get_torch_loss_cls(critic_loss_cls)()
        self.train_iters = train_iters
        self.actor_loss_coefficient = actor_loss_coefficient
        self.clip_ratio = clip_ratio


class ActorCritic(AbsAgent):
    """Actor Critic algorithm with separate policy and value models.

    References:
        https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch.
        https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f

    Args:
        model (SimpleMultiHeadModel): Multi-task model that computes action distributions and state values.
            It may or may not have a shared bottom stack.
        config: Configuration for the AC algorithm.
        experience_memory_size (int): Size of the experience memory. If it is -1, the experience memory is of
            unlimited size.
        experience_memory_overwrite_type (str): A string indicating how experiences in the experience memory are
            to be overwritten after its capacity has been reached. Must be "rolling" or "random".
        empty_experience_memory_after_step (bool): If True, the experience memory will be emptied  after each call
            to ``step``. Defaults to True.
        min_new_experiences_to_trigger_learning (int): Minimum number of new experiences required to trigger learning.
            Defaults to 1.
        min_experiences_to_trigger_learning (int): Minimum number of experiences in the experience memory required for training.
            Defaults to 1.
    """
    def __init__(
        self,
        model: SimpleMultiHeadModel,
        config: ActorCriticConfig,
        experience_memory_size: int,
        experience_memory_overwrite_type: str,
        empty_experience_memory_after_step: bool = True,
        min_new_experiences_to_trigger_learning: int = 1,
        min_experiences_to_trigger_learning: int = 1    
    ):
        if model.task_names is None or set(model.task_names) != {"actor", "critic"}:
            raise UnrecognizedTask(f"Expected model task names 'actor' and 'critic', but got {model.task_names}")
        super().__init__(
            model, config, experience_memory_size, experience_memory_overwrite_type,
            empty_experience_memory_after_step,
            min_new_experiences_to_trigger_learning=min_new_experiences_to_trigger_learning,
            min_experiences_to_trigger_learning=min_experiences_to_trigger_learning    
        )

    def choose_action(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Use the actor (policy) model to generate stochastic actions.

        Args:
            state: Input to the actor model.

        Returns:
            Actions and corresponding log probabilities.
        """
        state = torch.from_numpy(state).to(self.device)
        is_single = len(state.shape) == 1
        if is_single:
            state = state.unsqueeze(dim=0)

        action_prob = Categorical(self.model(state, task_name="actor", training=False))
        action = action_prob.sample()
        log_p = action_prob.log_prob(action)
        action, log_p = action.cpu().numpy(), log_p.cpu().numpy()
        return (action[0], log_p[0]) if is_single else (action, log_p)
    
    def step(self):
        batch = self.experience_memory.get()
        states = torch.from_numpy(np.asarray(batch["S"])).to(self.device)
        actions = torch.from_numpy(np.asarray([act[0] for act in batch["A"]])).to(self.device)
        log_p = torch.from_numpy(np.asarray([act[1] for act in batch["A"]])).to(self.device)
        rewards = torch.from_numpy(np.asarray(batch["R"])).to(self.device)
        next_states = torch.from_numpy(np.asarray(batch["S_"])).to(self.device)

        state_values = self.model(states, task_name="critic").detach().squeeze()
        next_state_values = self.model(next_states, task_name="critic").detach().squeeze()
        return_est = rewards + self.config.reward_discount * next_state_values
        advantages = return_est - state_values           
        for i in range(self.config.train_iters):
            # actor loss
            log_p_new = get_log_prob(self.model(states, task_name="actor"), actions)
            if self.config.clip_ratio is not None:
                ratio = torch.exp(log_p_new - log_p)
                clip_ratio = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
                actor_loss = -(torch.min(ratio * advantages, clip_ratio * advantages)).mean()
            else:
                actor_loss = -(log_p_new * advantages).mean()

            # critic_loss
            state_values = self.model(states, task_name="critic").squeeze()
            critic_loss = self.config.critic_loss_func(state_values, return_est)
            loss = critic_loss + self.config.actor_loss_coefficient * actor_loss

            self.model.step(loss)
