# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod

import torch
import torch.nn as nn


class AbsCoreModel(nn.Module):
    """Model abstraction for use in deep RL algorithms.

    This can be viewed as a container of one or more network components with embedded optimizers. This abstraction
    exposes simple and unified interfaces to decouple model inference and optimization from the algorithmic aspects
    of the policy that uses it.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def step(self, loss: torch.tensor):
        """Use a computed loss to back-propagate gradients and apply them to the underlying parameters.

        Args:
            loss: Result of a computation graph that involves the underlying parameters.
        """
        raise NotImplementedError

    def get_gradients(self, loss: torch.tensor):
        """Get gradients from a computed loss.

        There are two possible scenarios where you need to implement this interface: 1) if you are doing distributed
        learning and want each roll-out instance to collect gradients that can be directly applied to policy parameters
        on the learning side (abstracted through ``AbsPolicyManager``); 2) if you are computing loss in data-parallel
        fashion, i.e., by splitting a data batch to several smaller batches and sending them to a set of remote workers
        for parallelized gradient computation. In this case, this method will be used by the remote workers.
        """
        pass

    def apply_gradients(self, grad: dict):
        """Apply gradients to the model parameters.

        This needs to be implemented together with ``get_gradients``.
        """
        pass

    @abstractmethod
    def get_state(self):
        """Return the current model state.

        Ths model state usually involves the "state_dict" of the module as well as those of the embedded optimizers.
        """
        pass

    @abstractmethod
    def set_state(self, state):
        """Set model state.

        Args:
            state: Model state to be applied to the instance. Ths model state is either the result of a previous call
            to ``get_state`` or something loaded from disk and involves the "state_dict" of the module as well as those
            of the embedded optimizers.
        """
        pass
