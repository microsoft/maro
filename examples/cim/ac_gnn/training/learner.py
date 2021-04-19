# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from statistics import mean, stdev
from os import makedirs
from os.path import dirname, join, realpath

from maro.rl import AbsLearner

from examples.cim.ac_gnn.utils import batch_states, combine


class BasicLearner(AbsLearner):
    """Learner class for the training pipeline and the specialized logging in GNN solution for CIM problem."""
    def __init__(
        self, group_name, num_actors, max_ep, agent, logger,
        update_trigger=None, inference_trigger=None, train_freq=1, model_save_freq=1, proxy_options=None
    ):
        super().__init__(
            group_name, num_actors, agent,
            update_trigger=update_trigger, inference=True, inference_trigger=inference_trigger,
            state_batching_func=batch_states, proxy_options=proxy_options
        )
        self._max_ep = max_ep
        self._train_freq = train_freq
        self._model_save_freq = model_save_freq
        self._model_save_dir = join(dirname(dirname(realpath(__file__))), "model")
        makedirs(self._model_save_dir, exist_ok=True)
        self.logger = logger

    def run(self):
        rollout_time = training_time = 0
        for ep in range(self._max_ep):
            rollout_start = time.time()
            metrics_by_src, details_by_src = self.collect(ep)
            exp_by_src = {src: exp for src, (exp, logs) in details_by_src.items()}
            rollout_time += time.time() - rollout_start
            num_orders = list(metrics_by_src.values())[0]["order_requirements"]
            shortage_list = [met["container_shortage"] for met in metrics_by_src.values()]
            num_op_list = [met["operation_number"] for met in metrics_by_src.values()]
            st_mean, st_stdev = mean(shortage_list), stdev(shortage_list)
            op_mean, op_stdev = mean(num_op_list), stdev(num_op_list)
            self.logger.info(
                f"ep-{ep}: order={num_orders}, shortage={st_mean}({st_stdev}), num_op={op_mean}({op_stdev})"
            )
            train_start = time.time()
            self.agent.store_experiences(combine(exp_by_src))
            if ep % self._train_freq == self._train_freq - 1:
                self.agent.train()
                self.logger.debug("Training finished")
            training_time += time.time() - train_start
            if (ep + 1) % self._model_save_freq == 0:
                self.agent.dump_model_to_file(join(self._model_save_dir, str(ep)))

            self.logger.debug(f"rollout time: {int(rollout_time)}")
            self.logger.debug(f"training time: {int(training_time)}")
