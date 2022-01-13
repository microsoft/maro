from maro.rl_v3.learning import SimpleAgentWrapper
from maro.rl_v3.learning.rollout_manager import MultiProcessRolloutManager
from maro.rl_v3.tmp_example_single.config import algorithm, env_conf
from maro.rl_v3.tmp_example_single.env_sampler import CIMEnvSampler
from examples.rl.cim.ac import get_policy_func_dict
from maro.simulator import Env


def get_env_sampler_func() -> CIMEnvSampler:
    return CIMEnvSampler(
        get_env_func=lambda: Env(**env_conf),
        get_policy_func_dict=get_policy_func_dict,
        agent2policy={agent: f"{algorithm}.{agent}" for agent in Env(**env_conf).agent_idx_list},
        agent_wrapper_cls=SimpleAgentWrapper,
    )


if __name__ == "__main__":
    rollout_manager = MultiProcessRolloutManager(
        get_env_sampler_func=get_env_sampler_func,
        num_rollouts=5,
        num_steps=-1,
        num_eval_rollouts=3
    )
    # rollout_manager.collect(ep=1, segment=1, policy_state_dict=None)
