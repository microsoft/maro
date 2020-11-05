# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import itertools
from abc import ABC, abstractmethod
from typing import Union


class AbsAlgorithm(ABC):
    """Abstract RL algorithm class.

    The class provides uniform policy interfaces such as ``choose_action`` and ``train``. We also provide some
    predefined RL algorithm based on it, such DQN, A2C, etc. User can inherit from it to customize their own
    algorithms.

    Args:
        model_dict (dict): Underlying models for the algorithm (e.g., for A2C, model_dict could be something like
            {"actor": ..., "critic": ...})
        optimizer_opt (tuple or dict): Tuple or dict of tuples of (optimizer_class, optimizer_params) associated
            with the models in model_dict. If it is a tuple, the optimizer to be instantiated applies to all
            trainable parameters from ``model_dict``. If it is a dict, the optimizer will be applied to the related
            model with the same key.
        loss_func_dict (dict): Loss function types associated with the models.
        hyper_params (object): Algorithm-specific hyper-parameter set.
    """
    def __init__(self, model_dict: dict, optimizer_opt: Union[dict, tuple], loss_func_dict: dict, hyper_params: object):
        self._loss_func_dict = loss_func_dict
        self._hyper_params = hyper_params
        self._model_dict = model_dict
        self._register_optimizers(optimizer_opt)

    def _register_optimizers(self, optimizer_opt):
        if isinstance(optimizer_opt, tuple):
            # If a single optimizer_opt tuple is provided, a single optimizer will be created to jointly
            # optimize all model parameters involved in the algorithm.
            optim_cls, optim_params = optimizer_opt
            model_params = [model.parameters() for model in self._model_dict.values()]
            self._optimizer = optim_cls(itertools.chain(*model_params), **optim_params)
        else:
            self._optimizer = {}
            for model_key, model in self._model_dict.items():
                # No gradient required
                if model_key not in optimizer_opt or optimizer_opt[model_key] is None:
                    self._model_dict[model_key].eval()
                    self._optimizer[model_key] = None
                else:
                    optim_cls, optim_params = optimizer_opt[model_key]
                    self._optimizer[model_key] = optim_cls(model.parameters(), **optim_params)

    @property
    def model_dict(self):
        return self._model_dict

    @model_dict.setter
    def model_dict(self, model_dict):
        self._model_dict = model_dict

    @abstractmethod
    def train(self, *args, **kwargs):
        """Train models using samples.

        This method is algorithm-specific and needs to be implemented by the user. For example, for the DQN
        algorithm, this may look like train(self, state, action, reward, next_state).
        """
        return NotImplementedError

    @abstractmethod
    def choose_action(self, state, epsilon: float = None):
        """This method uses the underlying model(s) to compute an action from a shaped state.

        Args:
            state: A state object shaped by a ``StateShaper`` to conform to the model input format.
            epsilon (float, optional): Exploration rate. For greedy value-based algorithms, this being None means
                using the model output without exploration. For algorithms with inherently stochastic policies such
                as policy gradient, this is usually ignored. Defaults to None.

        Returns:
            The action to be taken given ``state``. It is usually necessary to use an ``ActionShaper`` to convert
            this to an environment executable action.
        """
        return NotImplementedError
