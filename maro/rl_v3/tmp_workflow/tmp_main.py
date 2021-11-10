import time
from typing import Callable, List

from maro.rl_v3.learning import AbsEnvSampler, AbsTrainerManager, ExpElement, SimpleAgentWrapper, SimpleTrainerManager
from maro.rl_v3.tmp_workflow.callbacks import cim_post_collect, cim_post_evaluate
from maro.rl_v3.tmp_workflow.config import env_conf
from maro.rl_v3.tmp_workflow.env_sampler import CIMEnvSampler
from maro.rl_v3.tmp_workflow.policies import get_policy_func_dict
from maro.rl_v3.tmp_workflow.trainers import get_trainer_func_dict, policy2trainer
from maro.simulator import Env


def main(
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
        print(f"\n========== Start of episode {ep} ==========")
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
            print(f"Huoran log: collect_time = {collect_time}")

            tu0 = time.time()
            trainer_manager.record_experiences(experiences)
            trainer_manager.train()
            policy_update_time += time.time() - tu0
            print(f"Huoran log: policy_update_time = {policy_update_time}")

            trainer_policy_states = trainer_manager.get_policy_states()
            policy_states = {}
            for v in trainer_policy_states.values():
                policy_states.update(v)

            tracker = env_sampler.test(policy_state_dict=policy_states)
            if post_evaluate:
                post_evaluate([tracker], ep)

            segment += 1

        env_sampler.test()


if __name__ == "__main__":
    algorithm = "dqn"
    main(
        get_env_sampler_func=lambda: CIMEnvSampler(
            get_env_func=lambda: Env(**env_conf),
            get_policy_func_dict=get_policy_func_dict,
            agent2policy={agent: f"{algorithm}.{agent}" for agent in Env(**env_conf).agent_idx_list},
            agent_wrapper_cls=SimpleAgentWrapper,
        ),
        get_trainer_manager_func=lambda: SimpleTrainerManager(
            get_trainer_func_dict=get_trainer_func_dict,
            get_policy_func_dict=get_policy_func_dict,
            agent2policy={agent: f"{algorithm}.{agent}" for agent in Env(**env_conf).agent_idx_list},
            policy2trainer=policy2trainer
        ),
        num_episodes=20,
        post_collect=cim_post_collect,
        post_evaluate=cim_post_evaluate
    )
