# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl.agent_manager import AbsAgentManager
from maro.rl.scheduling.scheduler import Scheduler
from maro.simulator import Env
from maro.utils import InternalLogger, Logger

from .abs_learner import AbsLearner


class SimpleLearner(AbsLearner):
    """A simple implementation of ``AbsLearner``.

    Args:
        env (Env): An Env instance.
        agent_manager (AbsAgentManager): An AgentManager instance that manages all agents.
        scheduler (AbsScheduler): A scheduler responsible for iterating over episodes and generating exploration
            parameters if necessary.
    """
    def __init__(
        self,
        env: Env,
        agent_manager: AbsAgentManager,
        scheduler: Scheduler,
        logger: Logger = InternalLogger("learner")
    ):
        super().__init__()
        self._env = env
        self.agent_manager = agent_manager
        self._scheduler = scheduler
        self._logger = logger

    def learn(self):
        """Main loop for collecting experiences from the actor and using them to update policies."""
        for exploration_params in self._scheduler:
            performance, exp_by_agent = self._sample(exploration_params=exploration_params)
            self._scheduler.record_performance(performance)
            ep_summary = f"ep {self._scheduler.current_ep} - performance: {performance}"
            if exploration_params:
                ep_summary = f"{ep_summary}, exploration_params: {self._scheduler.exploration_params}"
            self._logger.info(ep_summary)
            self.agent_manager.train(exp_by_agent)

    def test(self):
        """Test policy performance."""
        performance, _ = self._sample(return_details=False)
        self._scheduler.record_performance(performance)

    def load_models(self, dir_path: str):
        self.agent_manager.load_models_from_files(dir_path)

    def dump_models(self, dir_path: str):
        self.agent_manager.dump_models_to_files(dir_path)

    def _sample(self, exploration_params=None, return_details: bool = True):
        """Perform one episode of roll-out and return performance and experiences.

        Args:
            exploration_params: Exploration parameters.
            return_details (bool): If True, return experiences as well as performance metrics provided by the env.

        Returns:
            Performance and relevant details from the episode (e.g., experiences).
        """
        self._env.reset()

        # load exploration parameters:
        if exploration_params is not None:
            self.agent_manager.set_exploration_params(exploration_params)

        metrics, decision_event, is_done = self._env.step(None)
        while not is_done:
            action = self.agent_manager.choose_action(decision_event, self._env.snapshot_list)
            metrics, decision_event, is_done = self._env.step(action)
            self.agent_manager.on_env_feedback(metrics)

        details = self.agent_manager.post_process(self._env.snapshot_list) if return_details else None

        return self._env.metrics, details
