# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

from maro.rl import AbsAgent, ColumnBasedStore


class CIMAgent(AbsAgent):
    def train(self, ):
        for _ in range(self._num_batches):
            indexes, sample = self._experience_pool.sample_by_key("loss", self._batch_size)
            state = np.asarray(sample["state"])
            action = np.asarray(sample["action"])
            reward = np.asarray(sample["reward"])
            next_state = np.asarray(sample["next_state"])
            loss = self._algorithm.train(state, action, reward, next_state)
            self._experience_pool.update(indexes, {"loss": loss})
