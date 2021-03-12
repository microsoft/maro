# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from os import makedirs
from os.path import dirname, join, realpath

from maro.rl import AbsLearner
from maro.utils import DummyLogger

from examples.cim.ac_gnn.utils import batch_states, combine


class BasicLearner(AbsLearner):
    """Learner class for the training pipeline and the specialized logging in GNN solution for CIM problem."""
    def __init__(
        self, group_name, num_actors, max_ep, agent,
        update_trigger=None, inference_trigger=None, train_freq=1, model_save_freq=1, logger=None
    ):
        super().__init__(
            group_name, num_actors, agent,
            update_trigger=update_trigger, inference=True, inference_trigger=inference_trigger,
            state_batching_func=batch_states
        )
        self._max_ep = max_ep
        self._train_freq = train_freq
        self.model_save_freq = model_save_freq
        self.model_save_dir = join(dirname(dirname(realpath(__file__))), "model")
        makedirs(self.model_save_dir, exist_ok=True)
        self.logger = logger if logger else DummyLogger()

    def run(self):
        rollout_time = training_time = 0
        for ep in range(self._max_ep):
            rollout_start = time.time()
            metrics_by_src, details_by_src = self.collect(ep)
            exp_by_src = {src: exp for src, (exp, logs) in details_by_src.items()}
            rollout_time += time.time() - rollout_start
            for src, metrics in metrics_by_src.items():
                self.logger.info(f"{src}.ep-{ep} - performance: {metrics}")
            train_start = time.time()
            self.agent.store_experiences(combine(exp_by_src))
            if ep % self._train_freq == self._train_freq - 1:
                self.agent.learn()
                self.logger.info("Training finished")
            training_time += time.time() - train_start
            if (ep + 1) % self.model_save_freq == 0:
                self.agent.dump_model_to_file(join(self.model_save_dir, str(ep)))

            self.logger.debug(f"rollout time: {int(rollout_time)}")
            self.logger.debug(f"training time: {int(training_time)}")
