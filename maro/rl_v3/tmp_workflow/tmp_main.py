import time
from collections import Callable

from maro.rl_v3.learning import AbsEnvSampler, AbsTrainerManager


def main(
    get_env_sampler_func: Callable[[], AbsEnvSampler],
    get_trainer_manager_func: Callable[[], AbsTrainerManager],
    num_episodes: int,
    num_steps: int
):
    env_sampler = get_env_sampler_func()
    trainer_manager = get_trainer_manager_func()

    for ep in range(1, num_episodes + 1):
        collect_time = policy_update_time = 0
        segment = 1
        end_of_episode = False
        while not end_of_episode:
            tc0 = time.time()
            env_sampler.sample(num_steps=num_steps)
            collect_time += time.time() - tc0

            tu0 = time.time()
            trainer_manager.train()
            policy_update_time += time.time() - tu0

            segment += 1


if __name__ == '__main__':
    main()  # TODO
