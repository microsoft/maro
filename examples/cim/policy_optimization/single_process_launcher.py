# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from statistics import mean

import numpy as np

from maro.simulator import Env
from maro.rl import AgentManagerMode, Scheduler, SimpleActor, SimpleLearner
from maro.utils import LogFormat, Logger, convert_dottable

from components import CIMActionShaper, CIMStateShaper, POAgentManager, TruncatedExperienceShaper, create_po_agents


class EarlyStoppingChecker:
    """Callable class that checks the performance history to determine early stopping.

    Args:
        warmup_ep (int): Episode from which early stopping checking is initiated.
        last_k (int): Number of latest performance records to check for early stopping.
        perf_threshold (float): The mean of the ``last_k`` performance metric values must be above this value to
            trigger early stopping.
        perf_stability_threshold (float): The maximum one-step change over the ``last_k`` performance metrics must be
            below this value to trigger early stopping.
    """
    def __init__(self, warmup_ep: int, last_k: int, perf_threshold: float, perf_stability_threshold: float):
        self._warmup_ep = warmup_ep
        self._last_k = last_k
        self._perf_threshold = perf_threshold
        self._perf_stability_threshold = perf_stability_threshold

        def get_metric(record):
            return 1 - record["container_shortage"] / record["order_requirements"]
        self._metric_func = get_metric

    def __call__(self, perf_history) -> bool:
        if len(perf_history) < max(self._last_k, self._warmup_ep):
            return False

        metric_series = list(map(self._metric_func, perf_history[-self._last_k:]))
        max_delta = max(
            abs(metric_series[i] - metric_series[i - 1]) / metric_series[i - 1] for i in range(1, self._last_k)
        )
        print(f"mean_metric: {mean(metric_series)}, max_delta: {max_delta}")
        return mean(metric_series) > self._perf_threshold and max_delta < self._perf_stability_threshold


def launch(config):
    # First determine the input dimension and add it to the config.
    config = convert_dottable(config)

    # Step 1: initialize a CIM environment for using a toy dataset.
    env = Env(config.env.scenario, config.env.topology, durations=config.env.durations)
    agent_id_list = [str(agent_id) for agent_id in env.agent_idx_list]

    # Step 2: create state, action and experience shapers. We also need to create an explorer here due to the
    # greedy nature of the DQN algorithm.
    state_shaper = CIMStateShaper(**config.env.state_shaping)
    action_shaper = CIMActionShaper(action_space=list(np.linspace(-1.0, 1.0, config.agents.num_actions)))
    experience_shaper = TruncatedExperienceShaper(**config.env.experience_shaping)

    # Step 3: create an agent manager.
    config["agents"]["input_dim"] = state_shaper.dim
    agent_manager = POAgentManager(
        name="cim_learner",
        mode=AgentManagerMode.TRAIN_INFERENCE,
        agent_dict=create_po_agents(agent_id_list, config.agents),
        state_shaper=state_shaper,
        action_shaper=action_shaper,
        experience_shaper=experience_shaper,
    )

    # Step 4: Create an actor and a learner to start the training process.
    scheduler = Scheduler(
        config.main_loop.max_episode,
        early_stopping_checker=EarlyStoppingChecker(**config.main_loop.early_stopping)
    )
    actor = SimpleActor(env, agent_manager)
    learner = SimpleLearner(
        agent_manager, actor, scheduler,
        logger=Logger("cim_learner", format_=LogFormat.simple, auto_timestamp=False)
    )
    learner.learn()
    learner.test()
    learner.dump_models(os.path.join(os.getcwd(), "models"))


if __name__ == "__main__":
    from components.config import config
    launch(config)
