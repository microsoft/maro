# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable

import torch


def select_by_actions(q_values: torch.Tensor, actions: torch.Tensor):
    if len(actions.shape) == 1:
        actions = actions.unsqueeze(1)  # (N, 1)
    return q_values.gather(1, actions).squeeze(1)


def get_max(q_values: torch.Tensor, expand_action_dim: bool = True):
    """
    Given Q-values for a batch of states and all actions, return the maximum Q-value and
    the corresponding action index for each state.
    """
    greedy_q, actions = q_values.max(dim=1)
    if expand_action_dim:
        actions = actions.unsqueeze(1)
    return greedy_q, actions


def get_td_errors(
    q_values: torch.Tensor, next_q_values: torch.Tensor, rewards: torch.Tensor, gamma: float,
    loss_func: Callable
):
    target_q_values = (rewards + gamma * next_q_values).detach()  # (N,)
    return loss_func(q_values, target_q_values)


def get_log_prob(action_probs: torch.Tensor, actions: torch.Tensor):
    return torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())  # (N,)
