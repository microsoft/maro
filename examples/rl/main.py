# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from os.path import dirname, join, realpath 

from maro.rl.training import TrainingManager
from maro.rl.workflows.scenario import Scenario
from maro.utils import LoggerV2

# config variables
SCENARIO_NAME = "cim"
SCENARIO_PATH = join(dirname(dirname(realpath(__file__))), SCENARIO_NAME, "rl")
NUM_EPISODES = 50
NUM_STEPS = None
CHECKPOINT_PATH = join(dirname(SCENARIO_PATH), "checkpoints")
CHECKPOINT_INTERVAL = 5
EVAL_SCHEDULE = [10, 20, 30, 40, 50]
LOG_PATH = join(dirname(SCENARIO_PATH), "logs", SCENARIO_NAME)


if __name__ == "__main__":
    scenario = Scenario(SCENARIO_PATH)
    logger = LoggerV2("MAIN", dump_path=LOG_PATH)

    agent2policy = scenario.agent2policy
    policy_creator = scenario.policy_creator
    policy_dict = {name: get_policy_func(name) for name, get_policy_func in policy_creator.items()}
    policy_creator = {name: lambda name: policy_dict[name] for name in policy_dict}
    trainer_creator = scenario.trainer_creator

    # evaluation schedule
    logger.info(f"Policy will be evaluated at the end of episodes {EVAL_SCHEDULE}")
    eval_point_index = 0

    if scenario.trainable_policies is None:
        trainable_policies = set(policy_creator.keys())
    else:
        trainable_policies = set(scenario.trainable_policies)

    env_sampler = scenario.env_sampler_creator(policy_creator)

    trainable_policy_creator = {name: func for name, func in policy_creator.items() if name in trainable_policies}
    trainable_agent2policy = {id_: name for id_, name in agent2policy.items() if name in trainable_policies}
    training_manager = TrainingManager(
        trainable_policy_creator,
        trainer_creator,
        trainable_agent2policy,
        logger=logger
    )

    # main loop
    for ep in range(1, NUM_EPISODES + 1):
        collect_time = training_time = 0
        segment, end_of_episode = 1, False
        while not end_of_episode:
            # experience collection
            result = env_sampler.sample(num_steps=NUM_STEPS)
            experiences = result["experiences"]
            end_of_episode = result["end_of_episode"]

            if scenario.post_collect:
                scenario.post_collect(result["info"], ep, segment)

            logger.info(f"Roll-out completed for episode {ep}. Training started...")
            training_manager.record_experiences(experiences)
            training_manager.train_step()
            if CHECKPOINT_PATH and ep % CHECKPOINT_INTERVAL == 0:
                pth = join(CHECKPOINT_PATH, str(ep))
                training_manager.save(pth)
                logger.info(f"All trainer states saved under {pth}")
            segment += 1

        # performance details
        if ep == EVAL_SCHEDULE[eval_point_index]:
            eval_point_index += 1
            result = env_sampler.eval()
            if scenario.post_evaluate:
                scenario.post_evaluate(result["info"], ep)
