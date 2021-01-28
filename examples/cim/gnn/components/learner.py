import os
import time

from maro.rl import AbsLearner
from maro.utils import DummyLogger


class GNNLearner(AbsLearner):
    """Learner class for the training pipeline and the specialized logging in GNN solution for CIM problem."""
    def __init__(
        self, env, agent_manager, scheduler, train_freq=1, model_save_freq=1, log_pth=os.getcwd(), logger=DummyLogger()
    ):
        super().__init__()
        self._env = env
        self.agent_manager = agent_manager
        self._scheduler = scheduler
        self._train_freq = train_freq
        self._model_save_freq = model_save_freq
        self._log_pth = log_pth
        self._logger = logger

    def learn(self):
        rollout_time = 0
        training_time = 0
        for _ in self._scheduler:
            tick = time.time()
            performance, exp_dict = self._sample()
            rollout_time += time.time() - tick
            self._logger.info(f"ep {self._scheduler.current_ep} - performance: {performance}")
            self.agent_manager.store_experiences(exp_dict)

            if self._scheduler.current_ep % self._train_freq == self._train_freq - 1:
                self._logger.info("training start")
                tick = time.time()
                self.agent_manager.train()
                training_time += time.time() - tick

            if self._log_pth is not None and (self._scheduler.current_ep + 1) % self._model_save_freq == 0:
                self.agent_manager.dump_models_to_files(
                    os.path.join(self._log_pth, "models", self._scheduler.current_ep + 1)
                )

            self._logger.debug(f"total rollout_time: {int(rollout_time)}")
            self._logger.debug(f"train_time: {int(training_time)}")

    def _sample(self, return_details: bool = True):
        self._env.reset()
        metrics, decision_event, is_done = self._env.step(None)
        while not is_done:
            action = self.agent_manager.choose_action(decision_event, self._env.snapshot_list)
            metrics, decision_event, is_done = self._env.step(action)
            self.agent_manager.on_env_feedback(metrics)

        details = self.agent_manager.post_process(self._env.snapshot_list) if return_details else None

        return self._env.metrics, details
