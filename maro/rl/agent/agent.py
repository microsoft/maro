# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pickle
from typing import Callable
import numpy as np
import torch

from maro.rl.storage.unbounded_store import UnboundedStore
from maro.rl.storage.fixed_size_store import FixedSizeStore, OverwriteType
from maro.rl.common import ExperienceKey, ExperienceInfoKey


class AgentParameters:
    def __init__(self, min_experiences_to_train: int, samplers: [Callable], num_steps: int,
                 store_capacity: int = None,
                 overwrite_type: OverwriteType = None,
                 index_sampler: Callable = None):
        """
        Parameters external to an agent's underlying algorithm
        Args:
            min_experiences_to_train (int): minimum number of experiences required for training.
            samplers ([Callable]): list of (store_key, weight_fn, sample_size) that specifies how to obtain training  \n
                                   samples from the experience pool.
            num_steps (int): number of training steps.
            store_capacity (int): capacity of the agent's experience store (ignored if store type = UNBOUNDED).
            overwrite_type (OverwriteType): an enum specifying how existing entries in the experience store should \n
                                            be overwritten when the store reaches capacity (ignored is store type = FIXED).
            index_sampler (Callable): a sampling function used to find overwrite positions according to a certain   \n
                                      rule. If store type = FIXED and this is not None, overwrite_type is ignored.
        """
        self.min_experiences_to_train = min_experiences_to_train
        self.samplers = samplers
        self.num_steps = num_steps
        self.store_capacity = store_capacity
        self.overwrite_type = overwrite_type
        self.index_sampler = index_sampler


class Agent(object):
    def __init__(self,
                 name: str,
                 algorithm,
                 params: AgentParameters):
        """
        RL agent class. One of the two most important abstractions in MARO (the other being Env).
        The centerpiece of an agent instance is the algorithm, which is responsible for choosing
        actions and optimizing models.
        Args:
            name (str): agent's name.
            algorithm: a concrete algorithm instance that inherits from AbstractAlgorithm. This is the centerpiece \n
                       of the Agent class and is responsible for the most important tasks of an agent: choosing    \n
                       actions and optimizing models.
            params: a collection of hyper-parameters associated with the model training loop.
        """
        self._name = name
        self._algorithm = algorithm
        self._params = params

        def get_store():
            if self._params.store_capacity is None:
                return UnboundedStore()
            else:
                return FixedSizeStore(capacity=self._params.store_capacity,
                                      overwrite_type=self._params.overwrite_type)

        self._experience_store = {**{key: get_store() for key in ExperienceKey}, **{"info": get_store()}}

    @property
    def algorithm(self):
        return self._algorithm

    def choose_action(self, model_state, epsilon: float = .0):
        """
        Choose an action using the underlying algorithm based a preprocessed env state.
        Args:
            model_state: state vector as accepted by the underlying algorithm.
            epsilon (float): exploration rate.
        Returns:
            Action given by the underlying policy model.
        """
        return self._algorithm.choose_action(model_state, epsilon)

    def train(self):
        """
        Runs a specified number of training steps, with each step consisting of sampling a batch from the experience  \n
        pool and running the underlying algorithm's train_on_batch() method.
        """
        size = next(iter(self._experience_store.values())).size
        if size < self._params.min_experiences_to_train:
            return

        for _ in range(self._params.num_steps):
            indexes, batch = self._sample()
            loss = self._algorithm.train_on_batch(batch)
            # update TD errors
            self._experience_store["info"].update(indexes, loss, key=ExperienceInfoKey.TD_ERROR)

    def store_experiences(self, experience_dict: dict):
        size = len(next(iter(experience_dict.values())))
        for key, lst in experience_dict.items():
            assert len(lst) == size, f"expected a list of length {size} for key {key}, got {len(lst)}"
            self._experience_store[key].put(lst)

    def load_model_dict(self, model_dict: dict):
        self._algorithm.model_dict = model_dict

    def load_model_dict_from(self, dir_path):
        for model_key, state_dict in torch.load(dir_path).items():
            self._algorithm.model_dict[model_key].load_state_dict(state_dict)

    def dump_model_dict(self, dir_path: str):
        torch.save({model_key: model.state_dict() for model_key, model in self._algorithm.model_dict.items()},
                   os.path.join(dir_path, self._name))

    def dump_experience_store(self, dir_path: str):
        with open(os.path.join(dir_path, self._name)) as fp:
            pickle.dump(self._experience_store, fp)
    
    def _sample(self):
        indexes, info_items = self._experience_store["info"].apply_multi_samplers(self._params.samplers)
        batch = {ExperienceInfoKey.DISCOUNT: np.asarray([x[ExperienceInfoKey.DISCOUNT] for x in info_items])}
        for key in ExperienceKey:
            batch[key] = np.vstack(self._experience_store[key].get(indexes))

        return indexes, batch
