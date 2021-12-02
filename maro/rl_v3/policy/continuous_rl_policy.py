from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from maro.rl_v3.model import ContinuousPolicyNet
from .abs_policy import RLPolicy


def _parse_action_range(
    action_dim: int,
    action_range: Tuple[Union[float, List[float]], Union[float, List[float]]]
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
    def __init__(
        self,
        name: str,
        action_range: Tuple[Union[float, List[float]], Union[float, List[float]]],
        policy_net: ContinuousPolicyNet,
        trainable: bool = True
    ) -> None:
        """
        Args:
            name (str): Name of the policy.
            action_range: Value range of actions. Both the lower bound and the upper bound could be float or array. If
                it is an array, it should contain the bound for every dimension. If it is a float, it will be
                broadcast to all dimensions.
            policy_net (ContinuousPolicyNet): The core net of this policy.
            trainable (bool): Whether this policy is trainable. Defaults to True.
        """
        assert isinstance(policy_net, ContinuousPolicyNet)

        super(ContinuousRLPolicy, self).__init__(
            name=name, state_dim=policy_net.state_dim, action_dim=policy_net.action_dim,
            trainable=trainable
        )

        self._lbounds, self._ubounds = _parse_action_range(self.action_dim, action_range)
        assert self._lbounds is not None and self._ubounds is not None

        self._policy_net = policy_net

    @property
    def action_bounds(self) -> Tuple[List[float], List[float]]:
        return self._lbounds, self._ubounds

    @property
    def policy_net(self) -> ContinuousPolicyNet:
        return self._policy_net

    def _post_check(self, states: torch.Tensor, actions: torch.Tensor) -> bool:
        return all([
            (np.array(self._lbounds) <= actions.cpu().numpy()).all(),
            (actions.cpu().numpy() < np.array(self._ubounds)).all()
        ])

    def _get_actions_impl(self, states: torch.Tensor, exploring: bool) -> torch.Tensor:
        return self._policy_net.get_actions(states, exploring)

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

    def get_policy_state(self) -> object:
        return self._policy_net.get_net_state()

    def set_policy_state(self, policy_state: object) -> None:
        self._policy_net.set_net_state(policy_state)

    def soft_update(self, other_policy: RLPolicy, tau: float) -> None:
        assert isinstance(other_policy, ContinuousRLPolicy)
        self._policy_net.soft_update(other_policy.policy_net, tau)

    def _to_device_impl(self, device: torch.device) -> None:
        self._policy_net.to(device)
