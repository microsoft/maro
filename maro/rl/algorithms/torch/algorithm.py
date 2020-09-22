# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from typing import Union
import itertools


class Algorithm(object):
    def __init__(self, model_dict: dict, optimizer_opt: Union[dict, tuple], loss_func_dict: dict, hyper_params):
        """
        This represents MARO's support for RL algorithms, such as DQN and A2C. User-defined algorithms must
        inherit from this and provide concrete implementations of all the abstract methods here.
        Args:
            model_dict (dict): underlying models for the algorithm (e.g., for A2C,
                               model_dict = {"actor": ..., "critic": ...})
            optimizer_opt (tuple or dict): tuple or dict of tuples of (optimizer_class, optimizer_params) associated
                                           with the models in model_dict. If it is a tuple, the optimizer to be
                                           instantiated applies to all trainable parameters from model_dict. If it
                                           is a dict, it must have the same keys as model_dict.
            loss_func_dict (dict): loss function types associated with the models in model_dict.
            hyper_params: algorithm-specific hyper-parameter set
        """
        self._optimizer_opt = optimizer_opt
        self._loss_func_dict = loss_func_dict
        self._hyper_params = hyper_params
        self._model_dict = model_dict
        self._register_optimizers()

    def _register_optimizers(self):
        # TODO: single optimizer -> joint optimizing for all sub modules
        if isinstance(self._optimizer_opt, tuple):
            optim_cls, optim_params = self._optimizer_opt
            model_params = [model.parameters() for model in self._model_dict.values()]
            self._optimizer = optim_cls(itertools.chain(*model_params), **optim_params)
        else:
            self._optimizer = {}
            for model_key, model in self._model_dict.items():
                # no gradient required
                if model_key not in self._optimizer_opt or self._optimizer_opt[model_key] is None:
                    self._model_dict[model_key].eval()
                    self._optimizer[model_key] = None
                else:
                    optim_cls, optim_params = self._optimizer_opt[model_key]
                    self._optimizer[model_key] = optim_cls(model.parameters(), **optim_params)

    @property
    def model_dict(self):
        return self._model_dict

    @model_dict.setter
    def model_dict(self, model_dict):
        self._model_dict = model_dict

    @abstractmethod
    def choose_action(self, state, epsilon: float = None):
        return NotImplementedError

    @abstractmethod
    def train_on_batch(self, batch):
        return NotImplementedError
