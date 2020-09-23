# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

from maro.rl import AbsAgent, ColumnBasedStore


class ECRAgent(AbsAgent):
    def __init__(self, name, algorithm, experience_pool: ColumnBasedStore, min_experiences_to_train,
                 num_batches, batch_size):
        super().__init__(name, algorithm, experience_pool)
        self._min_experiences_to_train = min_experiences_to_train
        self._num_batches = num_batches
        self._batch_size = batch_size

    def train(self):
        if len(self._experience_pool) < self._min_experiences_to_train:
            return

        for _ in range(self._num_batches):
            indexes, sample = self._experience_pool.sample_by_key("loss", self._batch_size)
            state = np.asarray(sample["state"])
            action = np.asarray(sample["action"])
            reward = np.asarray(sample["reward"])
            next_state = np.asarray(sample["next_state"])
            loss = self._algorithm.train(state, action, reward, next_state)
            self._experience_pool.update(indexes, {"loss": loss})
