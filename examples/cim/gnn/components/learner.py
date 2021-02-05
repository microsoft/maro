import os
import time

from maro.rl import AbsLearner
from maro.utils import DummyLogger


class Learner(AbsLearner):
    """Learner class for the training pipeline and the specialized logging in GNN solution for CIM problem."""
    def __init__(
        self, actor, scheduler, train_freq=1, model_save_freq=1, log_pth=os.getcwd(), logger=DummyLogger()
    ):
        super().__init__(actor, scheduler)
        self._train_freq = train_freq
        self._model_save_freq = model_save_freq
        self._log_pth = log_pth
        self._logger = logger

    def learn(self):
        rollout_time = 0
        training_time = 0
        for _ in self.scheduler:
            rollout_start = time.time()
            exp, logs = self.actor.roll_out(self.scheduler.iter)
            rollout_time += time.time() - rollout_start
            self._logger.info(f"ep {self.scheduler.iter} - performance: {self.actor.env.metrics}")
            train_start = time.time()
            self.update(exp)
            training_time += time.time() - train_start
            if self._log_pth is not None and (self.scheduler.iter + 1) % self._model_save_freq == 0:
                self.actor.agent.dump_model_to_file(
                    os.path.join(self._log_pth, "models", str(self.scheduler.iter + 1))
                )

            self._logger.debug(f"rollout time: {int(rollout_time)}")
            self._logger.debug(f"training time: {int(training_time)}")

    def update(self, exp_dict):
        self.actor.agent.store_experiences(exp_dict)
        if self.scheduler.iter % self._train_freq == self._train_freq - 1:
            self._logger.info("training start")
            self.actor.agent.train()
