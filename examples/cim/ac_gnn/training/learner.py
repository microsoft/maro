# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from collections import defaultdict
from os import makedirs
from os.path import dirname, join, realpath

import numpy as np

from maro.rl import AbsLearner
from maro.utils import DummyLogger

from examples.cim.ac_gnn.utils import batch_states, combine

from .numpy_store import Shuffler


class BasicLearner(AbsLearner):
    """Learner class for the training pipeline and the specialized logging in GNN solution for the CIM problem."""
    def __init__(
        self, group_name, num_actors, max_ep, agent, experience_pool, train_iter, batch_size,
        update_trigger=None, inference_trigger=None, train_freq=1, model_save_freq=1, logger=None
    ):
        super().__init__(
            group_name, num_actors, agent,
            update_trigger=update_trigger, inference=True, inference_trigger=inference_trigger,
            state_batching_func=batch_states
        )
        self.experience_pool = experience_pool
        self.train_iter = train_iter  # Number of training iterations per round of training
        self.batch_size = batch_size  # Mini-batch size
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
            self.store_experiences(combine(exp_by_src))
            if ep % self._train_freq == self._train_freq - 1:
                self.update()
                self.logger.info("Training finished")
            training_time += time.time() - train_start
            if (ep + 1) % self.model_save_freq == 0:
                self.agent.dump_model_to_file(join(self.model_save_dir, str(ep)))

            self.logger.debug(f"rollout time: {int(rollout_time)}")
            self.logger.debug(f"training time: {int(training_time)}")
    
    def store_experiences(self, experiences):
        for code, exp_list in experiences.items():
            self.experience_pool[code].put(exp_list)

    def update(self):
        for (p_idx, v_idx), exp_pool in self.experience_pool.items():
            loss_dict = defaultdict(list)
            for _ in range(self.train_iter):
                shuffler = Shuffler(exp_pool, batch_size=self.batch_size)
                while shuffler.has_next():
                    batch = shuffler.next()
                    actor_loss, critic_loss, entropy_loss, tot_loss = self.agent.learn(
                        batch["s"], batch["a"], batch["R"], batch["s_"], p_idx, v_idx
                    )
                    loss_dict["actor"].append(actor_loss)
                    loss_dict["critic"].append(critic_loss)
                    loss_dict["entropy"].append(entropy_loss)
                    loss_dict["tot"].append(tot_loss)

            if loss_dict:
                a_loss = np.mean(loss_dict["actor"])
                c_loss = np.mean(loss_dict["critic"])
                e_loss = np.mean(loss_dict["entropy"])
                tot_loss = np.mean(loss_dict["tot"])
                self.logger.debug(
                    f"code: {p_idx}-{v_idx} \t actor: {float(a_loss)} \t critic: {float(c_loss)} \t entropy: {float(e_loss)} \
                    \t tot: {float(tot_loss)}")

            exp_pool.clear()
