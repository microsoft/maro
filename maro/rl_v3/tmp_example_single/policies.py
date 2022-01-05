import torch

from maro.rl.exploration import MultiLinearExplorationScheduler, epsilon_greedy
from maro.rl_v3.policy import DiscretePolicyGradient, ValueBasedPolicy
from maro.rl_v3.training.algorithms import DQN, DiscreteActorCritic, DiscreteActorCriticParams, DQNParams
from maro.rl_v3.workflow import preprocess_get_policy_func_dict

from .config import algorithm, running_mode
from .nets import MyActorNet, MyCriticNet, MyQNet

dqn_policy_conf = {
    "exploration_strategy": (epsilon_greedy, {"epsilon": 0.4}),
    "exploration_scheduling_options": [(
        "epsilon", MultiLinearExplorationScheduler, {
            "splits": [(2, 0.32)],
            "initial_value": 0.4,
            "last_ep": 5,
            "final_value": 0.0,
        }
    )],
    "warmup": 100
}
dqn_params = DQNParams(
    device="cpu",
    reward_discount=.0,
    update_target_every=5,
    num_epochs=10,
    soft_update_coef=0.1,
    double=False,
    replay_memory_capacity=10000,
    random_overwrite=False,
    batch_size=32,
)
ac_params = DiscreteActorCriticParams(
    device="cpu",
    get_v_critic_net_func=lambda: MyCriticNet(),
    reward_discount=.0,
    grad_iters=10,
    critic_loss_cls=torch.nn.SmoothL1Loss,
    min_logp=None,
    lam=.0,
)


# #####################################################################################################################
def get_value_based_policy(name: str) -> ValueBasedPolicy:
    return ValueBasedPolicy(name=name, q_net=MyQNet(), **dqn_policy_conf)


def get_discrete_policy_gradient(name: str) -> DiscretePolicyGradient:
    return DiscretePolicyGradient(name=name, policy_net=MyActorNet())


def get_dqn(name: str) -> DQN:
    return DQN(name=name, params=dqn_params)


def get_ac(name: str) -> DiscreteActorCritic:
    return DiscreteActorCritic(name=name, params=ac_params)


if algorithm == "dqn":
    get_policy_func = get_value_based_policy
    get_policy_func_dict = {
        f"{algorithm}_{i}.{i}": get_policy_func
        for i in range(4)
    }

    get_trainer_func_dict = {
        f"{algorithm}_{i}": get_dqn
        for i in range(4)
    }

elif algorithm == "ac":
    get_policy_func = get_discrete_policy_gradient
    get_policy_func_dict = {
        f"{algorithm}_{i}.{i}": get_policy_func
        for i in range(4)
    }

    get_trainer_func_dict = {
        f"{algorithm}_{i}": get_ac
        for i in range(4)
    }
else:
    raise ValueError
# #####################################################################################################################

get_policy_func_dict = preprocess_get_policy_func_dict(
    get_policy_func_dict, running_mode
)
