# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
import itertools
from typing import Union


class AbsAlgorithm(ABC):
    def __init__(self, model_dict: dict, optimizer_opt: Union[dict, tuple], loss_func_dict: dict, hyper_params: object):
        """
        It's the abstraction of RL algorithm, which provides a uniform policy interface, such choose_action, train_on_batch.
        We also provide some predefined RL algorithm based on it, such DQN, A2C, etc. User can inherit from it to customize
        their own algorithms.

        Args:
            model_dict (dict): underlying models for the algorithm (e.g., for A2C,
                               model_dict = {"actor": ..., "critic": ...})
            optimizer_opt (tuple or dict): tuple or dict of tuples of (optimizer_class, optimizer_params) associated
                                           with the models in model_dict. If it is a tuple, the optimizer to be
                                           instantiated applies to all trainable parameters from model_dict. If it
                                           is a dict, the optimizer will be applied to the related model with the same key.
            loss_func_dict (dict): loss function types associated with the models in model_dict.
            hyper_params (object): algorithm-specific hyper-parameter set.
        """
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
        return NotImplementedError

    @abstractmethod
    def choose_action(self, state, epsilon: float = None):
        return NotImplementedError
