from maro.rl_v3.learning import SimpleAgentWrapper
from maro.rl_v3.training import SimpleTrainerManager
from maro.rl_v3.workflow import run_workflow_centralized_mode
from maro.simulator import Env

from .callbacks import cim_post_collect, cim_post_evaluate
from .config import algorithm, env_conf, running_mode
from .env_sampler import CIMEnvSampler
from .policies import get_policy_func_dict, get_trainer_func_dict

if __name__ == "__main__":
    run_workflow_centralized_mode(
        get_env_sampler_func=lambda: CIMEnvSampler(
            get_env_func=lambda: Env(**env_conf),
            get_policy_func_dict=get_policy_func_dict,
            agent2policy={agent: f"{algorithm}_{agent}.{agent}" for agent in Env(**env_conf).agent_idx_list},
            agent_wrapper_cls=SimpleAgentWrapper,
            device="cpu"
        ),
        get_trainer_manager_func=lambda: SimpleTrainerManager(
            get_policy_func_dict=get_policy_func_dict,
            agent2policy={agent: f"{algorithm}_{agent}.{agent}" for agent in Env(**env_conf).agent_idx_list},
            get_trainer_func_dict=get_trainer_func_dict
        ),
        num_episodes=30,
        post_collect=cim_post_collect,
        post_evaluate=cim_post_evaluate,
        running_mode=running_mode
    )
