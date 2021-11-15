import time
from typing import Callable, List

from maro.rl_v3.learning import AbsEnvSampler, AbsTrainerManager, ExpElement


def run_workflow(
    get_env_sampler_func: Callable[[], AbsEnvSampler],
    get_trainer_manager_func: Callable[[], AbsTrainerManager],
    num_episodes: int,
    num_steps: int = -1,
    post_collect: Callable[[List[dict], int, int], None] = None,
    post_evaluate: Callable[[List[dict], int], None] = None
):
    env_sampler = get_env_sampler_func()
    trainer_manager = get_trainer_manager_func()

    for ep in range(1, num_episodes + 1):
        if ep != 1:
            print('\n')
        print(f"========== Start of episode {ep} ==========")
        collect_time = policy_update_time = 0
        segment = 1
        end_of_episode = False
        while not end_of_episode:
            tc0 = time.time()
            sample_result = env_sampler.sample(num_steps=num_steps)
            end_of_episode: bool = sample_result["end_of_episode"]
            experiences: List[ExpElement] = sample_result["experiences"]

            tracker: dict = sample_result["tracker"]
            if post_collect:
                post_collect([tracker], ep, segment)
            collect_time += time.time() - tc0

            tu0 = time.time()
            trainer_manager.record_experiences(experiences)
            trainer_manager.train()
            policy_update_time += time.time() - tu0

            print(f"Huoran log: collect_time = {collect_time}, policy_update_time = {policy_update_time}")

            trainer_policy_states = trainer_manager.get_policy_states()
            policy_states = {}
            for v in trainer_policy_states.values():
                policy_states.update(v)

            env_sampler.set_policy_states(policy_states)
            # tracker = env_sampler.test(policy_state_dict=policy_states)
            # if post_evaluate:
            #     post_evaluate([tracker], ep)

            segment += 1
        print(f"========== End of episode {ep} ==========")