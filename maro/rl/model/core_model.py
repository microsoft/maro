# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.distributions import Categorical

from maro.rl.utils import get_torch_lr_scheduler_cls, get_torch_optim_cls
from maro.utils import clone
from maro.utils.exception.rl_toolkit_exception import MissingOptimizer


class OptimOption:
    """Model optimization options.
    Args:
        optim_cls: A string indicating an optimizer class provided by torch.optim or custom subclass of
            torch.optim.Optimizer. If a string is provided, it must be present in the ``TORCH_OPTIM`` index.
        optim_params (dict): Parameters for the optimizer class.
        scheduler_cls: A string indicating an lr-scheduler class provided by torch.optim.lr_scheduler or custom
            subclass of torch.optim.lr_scheduler. If a string is provided, it must be present in the
            ``TORCH_LR_SCHEDULER`` index. Defaults to None.
        scheduler_params (dict): Parameters for the scheduler class. Defaults to None.
    """
    __slots__ = ["optim_cls", "optim_params", "scheduler_cls", "scheduler_params"]

    def __init__(self, optim_cls, optim_params: dict, scheduler_cls=None, scheduler_params: dict = None):
        self.optim_cls = get_torch_optim_cls(optim_cls)
        self.optim_params = optim_params
        self.scheduler_cls = get_torch_lr_scheduler_cls(scheduler_cls)
        self.scheduler_params = scheduler_params


class AbsCoreModel(nn.Module):
    """Trainable model that consists of multiple network components.

    Args:
        component (Union[nn.Module, Dict[str, nn.Module]]): Network component(s) comprising the model.
        optim_option (Union[OptimOption, Dict[str, OptimOption]]): Optimizer options for the components.
            If none, no optimizer will be created for the model which means the model is not trainable.
            If it is a OptimOption instance, a single optimizer will be created to jointly optimize all
            parameters of the model. If it is a dictionary of OptimOptions, the keys will be matched against
            the component names and optimizers created for them. Note that it is possible to freeze certain
            components while optimizing others by providing a subset of the keys in ``component``.
            Defaults toNone.
    """
    def __init__(
        self,
        component: Union[nn.Module, Dict[str, nn.Module]],
        optim_option: Union[OptimOption, Dict[str, OptimOption]] = None,
        device: str = "cpu"
    ):
        super().__init__()
        self.component = component if isinstance(component, nn.Module) else nn.ModuleDict(component)
        if optim_option is None:
            self.optimizer = None
            self.scheduler = None
            self.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            if isinstance(optim_option, dict):
                self.optimizer = {}
                for name, opt in optim_option.items():
                    self.optimizer[name] = opt.optim_cls(self.component[name].parameters(), **opt.optim_params)
                    if opt.scheduler_cls:
                        self.scheduler[name] = opt.scheduler_cls(self.optimizer[name], **opt.scheduler_params)
            else:
                self.optimizer = optim_option.optim_cls(self.parameters(), **optim_option.optim_params)
                if optim_option.scheduler_cls:
                    self.scheduler = optim_option.scheduler_cls(self.optimizer, **optim_option.scheduler_params)

        self.device = torch.device(device)
        self.to(self.device)

    @property
    def trainable(self) -> bool:
        return self.optimizer is not None
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def step(self, loss):
        """Use the loss to back-propagate gradients and apply them to the underlying parameters."""
        if self.optimizer is None:
            raise MissingOptimizer("No optimizer registered to the model")
        if isinstance(self.optimizer, dict):
            for optimizer in self.optimizer.values():
                optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

        # Obtain gradients through back-propagation
        loss.backward()

        # Apply gradients
        if isinstance(self.optimizer, dict):
            for optimizer in self.optimizer.values():
                optimizer.step()
        else:
            self.optimizer.step()

    def update_learning_rate(self, component_name: Union[str, List[str]] = None):
        if not isinstance(self.scheduler, dict):
            self.scheduler.step()
        elif isinstance(component_name, str):
            if component_name not in self.scheduler:
                raise KeyError(f"Component {component_name} does not have a learning rate scheduler")
            self.scheduler[component_name].step()
        elif isinstance(component_name, list):
            for key in component_name:
                if key not in self.scheduler:
                    raise KeyError(f"Component {key} does not have a learning rate scheduler")
                self.scheduler[key].step()
        else:
            for sch in self.scheduler.values():
                sch.step()

    def soft_update(self, other_model: nn.Module, tau: float):
        for params, other_params in zip(self.parameters(), other_model.parameters()):
            params.data = (1 - tau) * params.data + tau * other_params.data

    def copy(self, with_optimizer: bool = False, device: str = None):
        model_copy = clone(self)
        if not with_optimizer:
            model_copy.optimizer = None
            model_copy.scheduler = None

        device = self.device if device is None else torch.device(device)
        model_copy.to(device)

        return model_copy


class DiscreteQNet(AbsCoreModel):
    """NN-based Q-value model."""
    @abstractmethod
    def forward(self, states) -> torch.tensor:
        """Output Q values"""
        raise NotImplementedError

    def get_action(self, states):
        """
        Given Q-values for a batch of states and all actions, return the maximum Q-value and
        the corresponding action index for each state.
        """
        q_for_all_actions = self.forward(states)  # (batch_size, num_actions)
        greedy_q, actions = q_for_all_actions.max(dim=1)
        return actions.detach(), greedy_q.detach()

    def q_values(self, states, actions: torch.tensor):
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(dim=1)
        q_for_all_actions = self.forward(states)  # (batch_size, num_actions)
        return q_for_all_actions.gather(1, actions).squeeze(dim=1)


class DiscretePolicyNet(AbsCoreModel):
    """A compound network structure that consists of multiple task heads and an optional shared stack.

    Args:
        component (Union[nn.Module, Dict[str, nn.Module]]): Network component(s) comprising the model.
            All components must have the same input dimension except the one designated as the shared
            component by ``shared_component_name``.
        optim_option (Union[OptimOption, Dict[str, OptimOption]]): Optimizer option for
            the components. Defaults to None.
    """
    @abstractmethod
    def forward(self, states) -> torch.tensor:
        raise NotImplementedError

    def get_action(self, states):
        """
        Given Q-values for a batch of states and all actions, return the maximum Q-value and
        the corresponding action index for each state.
        """
        action_prob = Categorical(self.forward(states))  # (batch_size, num_actions)
        action = action_prob.sample()
        log_p = action_prob.log_prob(action)
        return action, log_p


class DiscreteACNet(AbsCoreModel):
    """A compound network structure that consists of multiple task heads and an optional shared stack.

    Args:
        component (Union[nn.Module, Dict[str, nn.Module]]): Network component(s) comprising the model.
            All components must have the same input dimension except the one designated as the shared
            component by ``shared_component_name``.
        optim_option (Union[OptimOption, Dict[str, OptimOption]]): Optimizer option for
            the components. Defaults to None.
    """
    @abstractmethod
    def forward(self, states, output_action_probs: bool = True, output_values: bool = True):
        raise NotImplementedError

    def get_action(self, states):
        """
        Given Q-values for a batch of states, return the action index and the corresponding maximum Q-value
        for each state.
        """
        action_prob = Categorical(self.forward(states, output_values=False))  # (batch_size, num_actions)
        action = action_prob.sample()
        log_p = action_prob.log_prob(action)
        return action, log_p


class ContinuousACNet(AbsCoreModel):
    """A compound network structure that consists of multiple task heads and an optional shared stack.

    Args:
        component (Union[nn.Module, Dict[str, nn.Module]]): Network component(s) comprising the model.
            All components must have the same input dimension except the one designated as the shared
            component by ``shared_component_name``.
        optim_option (Union[OptimOption, Dict[str, OptimOption]]): Optimizer option for the components.
            Defaults to None.
    """
    @abstractmethod
    def forward(self, states, actions=None):
        raise NotImplementedError

    def get_action(self, states):
        """Compute the actions given a batch of states."""
        return self.forward(states)

    def value(self, states):
        """Compute the Q-values for a batch of states using the actions computed from them."""
        return self.forward(states, actions=self.get_action(states))
