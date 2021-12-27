# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import time

from maro.rl_v3.learning.helpers import get_rollout_finish_msg
from maro.rl_v3.policy_trainer.abs_trainer import BatchTrainer
from maro.rl_v3.utils.common import from_env, get_eval_schedule, get_logger, get_module


if __name__ == "__main__":
    # get user-defined scenario ingredients
    scenario = get_module(from_env("SCENARIO_PATH"))
    get_env_sampler = getattr(scenario, "get_env_sampler")
    get_train_ops_func_dict = getattr(scenario, "get_train_ops_func_dict")
    post_collect = getattr(scenario, "post_collect", None)
    post_evaluate = getattr(scenario, "post_evaluate", None)

    mode = from_env("MODE")
    num_episodes = from_env("NUM_EPISODES")
    num_steps = from_env("NUM_STEPS", required=False, default=-1)

    load_path = from_env("LOAD_PATH", required=False, default=None)
    checkpoint_path = from_env("CHECKPOINT_PATH", required=False, default=None)
    logger = get_logger(from_env("LOG_PATH", required=False, default=os.getcwd()), from_env("JOB"), "MAIN")

    # evaluation schedule
    eval_schedule = get_eval_schedule(from_env("EVAL_SCHEDULE", required=False, default=None), num_episodes)
    logger.info(f"Policy will be evaluated at the end of episodes {eval_schedule}")
    eval_point_index = 0
    if mode == "single":
        if checkpoint_path:
            os.makedirs(checkpoint_path, exist_ok=True)
        env_sampler = get_env_sampler()
        if load_path:
            env_sampler.agent_wrapper.load(load_path)
            logger.info(f"Loaded policy states from {load_path}")

        for ep in range(1, num_episodes + 1):
            collect_time = policy_update_time = 0
            segment, end_of_episode = 1, False
            while not end_of_episode:
                # experience collection
                tc0 = time.time()
                result = env_sampler.sample(num_steps=num_steps, return_rollout_info=False)
                trackers = [result["tracker"]]
                logger.info(
                    get_rollout_finish_msg(ep, result["step_range"], exploration_params=result["exploration_params"])
                )
                end_of_episode = result["end_of_episode"]

                if post_collect:
                    post_collect(trackers, ep, segment)

                collect_time += time.time() - tc0
                tu0 = time.time()
                env_sampler.agent_wrapper.improve()
                if checkpoint_path:
                    env_sampler.agent_wrapper.save(checkpoint_path)
                    logger.info(f"Saved policy states to {checkpoint_path}")
                policy_update_time += time.time() - tu0
                segment += 1

            # performance details
            logger.info(f"ep {ep} summary - collect time: {collect_time}, policy update time: {policy_update_time}")
            if eval_schedule and ep == eval_schedule[eval_point_index]:
                eval_point_index += 1
                trackers = [env_sampler.test()]
                if post_evaluate:
                    post_evaluate(trackers, ep)
    else:
        from rollout_manager import get_rollout_manager
        rollout_manager = get_rollout_manager()
        train_mode = from_env("TRAINING_MODE", required=True)
        if train_mode not in {"simple", "parallel"}:
            raise ValueError(f"Unsupported training mode: {train_mode}. Supported modes: simple, parallel")

        # Batching trainers for parallelism
        if train_mode == "parallel":
            dispatcher_address = (from_env("DISPATCHER_HOST"), from_env("DISPATCHER_PORT"))
        else:
            dispatcher_address = None
        batch_trainer = BatchTrainer(
            [
                spec["type"](
                    name,
                    module_path=from_env("SCENARIO_PATH"),
                    dospatcher_address=dispatcher_address,
                    **spec["parameters"]
                )
                for name, spec in getattr(scenario, "trainer_specs").items()
            ]
        )

        # main loop
        for ep in range(1, num_episodes + 1):
            collect_time = training_time = 0
            rollout_manager.reset()
            segment, end_of_episode = 1, False
            while not end_of_episode:
                # experience collection
                tc0 = time.time()
                policy_state_dict = batch_trainer.get_state()
                rollout_data, trackers = rollout_manager.collect(ep, segment, policy_state_dict)
                end_of_episode = rollout_manager.end_of_episode

                if post_collect:
                    post_collect(trackers, ep, segment)

                collect_time += time.time() - tc0
                tu0 = time.time()
                batch_trainer.record(rollout_data)
                batch_trainer.train()
                training_time += time.time() - tu0
                segment += 1

            # performance details
            logger.info(f"ep {ep} summary - collect time: {collect_time}, policy update time: {training_time}")
            if eval_schedule and ep == eval_schedule[eval_point_index]:
                eval_point_index += 1
                trackers = rollout_manager.evaluate(ep, batch_trainer.get_state())
                if post_evaluate:
                    post_evaluate(trackers, ep)

        rollout_manager.exit()


# # registry:
# {
#     "dqn.ops": some_fn,
#     "ac.ops": some_Fn,
#     "dqn2.ops": ...,
#     "maddpg.actor0": some_Fn,
#     "maddpg.actor1": some_fn,
#     ....

# }

# agent2policy = {"agent0": "maddpg.actor1"}
