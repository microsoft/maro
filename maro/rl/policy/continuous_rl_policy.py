# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from maro.rl.model import ContinuousPolicyNet

from .abs_policy import RLPolicy


def _parse_action_range(
    action_dim: int,
    action_range: Tuple[Union[float, List[float]], Union[float, List[float]]],
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    lower, upper = action_range

    if isinstance(lower, float):
        lower = [lower] * action_dim
    if isinstance(upper, float):
        upper = [upper] * action_dim

    if not (action_dim == len(lower) == len(upper)):
        return None, None

    for lval, uval in zip(lower, upper):
        if lval >= uval:
            return None, None

    return lower, upper


class ContinuousRLPolicy(RLPolicy):
    """RL policy for continuous action spaces.

    Args:
        name (str): Name of the policy.
        action_range (Tuple[Union[float, List[float]], Union[float, List[float]]]): Value range of actions.
            Both the lower bound and the upper bound could be float or array. If it is an array, it should contain
            the bound for every dimension. If it is a float, it will be broadcast to all dimensions.
        policy_net (ContinuousPolicyNet): The core net of this policy.
        trainable (bool, default=True): Whether this policy is trainable.
        warmup (int, default=0): Number of steps for uniform-random action selection, before running real policy.
            Helps exploration.
    """

    def __init__(
        self,
        name: str,
        action_range: Tuple[Union[float, List[float]], Union[float, List[float]]],
        policy_net: ContinuousPolicyNet,
        trainable: bool = True,
        warmup: int = 0,
    ) -> None:
        assert isinstance(policy_net, ContinuousPolicyNet)

        super(ContinuousRLPolicy, self).__init__(
            name=name,
            state_dim=policy_net.state_dim,
            action_dim=policy_net.action_dim,
            trainable=trainable,
            is_discrete_action=False,
            warmup=warmup,
        )

        self._lbounds, self._ubounds = _parse_action_range(self.action_dim, action_range)
        self._policy_net = policy_net

    @property
    def action_bounds(self) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        return self._lbounds, self._ubounds

    @property
    def policy_net(self) -> ContinuousPolicyNet:
        return self._policy_net

    def _post_check(self, states: torch.Tensor, actions: torch.Tensor, **kwargs) -> bool:
        return all(
            [
                (np.array(self._lbounds) <= actions.detach().cpu().numpy()).all(),
                (actions.detach().cpu().numpy() < np.array(self._ubounds)).all(),
            ],
        )

    def _get_actions_impl(self, states: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._policy_net.get_actions(states, self._is_exploring, **kwargs)

    def _get_random_actions_impl(self, states: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._policy_net.get_random_actions(states, **kwargs)

    def _get_actions_with_probs_impl(self, states: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._policy_net.get_actions_with_probs(states, self._is_exploring, **kwargs)

    def _get_actions_with_logps_impl(self, states: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._policy_net.get_actions_with_logps(states, self._is_exploring, **kwargs)

    def _get_states_actions_probs_impl(self, states: torch.Tensor, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._policy_net.get_states_actions_probs(states, actions, **kwargs)

    def _get_states_actions_logps_impl(self, states: torch.Tensor, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._policy_net.get_states_actions_logps(states, actions, **kwargs)

    def train_step(self, loss: torch.Tensor) -> None:
        self._policy_net.step(loss)

    def get_gradients(self, loss: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._policy_net.get_gradients(loss)

    def apply_gradients(self, grad: dict) -> None:
        self._policy_net.apply_gradients(grad)

    def freeze(self) -> None:
        self._policy_net.freeze()

    def unfreeze(self) -> None:
        self._policy_net.unfreeze()

    def eval(self) -> None:
        self._policy_net.eval()

    def train(self) -> None:
        self._policy_net.train()

    def get_state(self) -> dict:
        return {
            "net": self._policy_net.get_state(),
            "policy": {
                "warmup": self._warmup,
                "call_count": self._call_count,
            },
        }

    def set_state(self, policy_state: dict) -> None:
        self._policy_net.set_state(policy_state["net"])
        self._warmup = policy_state["policy"]["warmup"]
        self._call_count = policy_state["policy"]["call_count"]

    def soft_update(self, other_policy: RLPolicy, tau: float) -> None:
        assert isinstance(other_policy, ContinuousRLPolicy)
        self._policy_net.soft_update(other_policy.policy_net, tau)

    def _to_device_impl(self, device: torch.device) -> None:
        self._policy_net.to_device(device)
