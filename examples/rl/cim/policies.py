import torch

from maro.rl_v3.exploration import MultiLinearExplorationScheduler, epsilon_greedy
from maro.rl_v3.policy import DiscretePolicyGradient, ValueBasedPolicy
from maro.rl_v3.training.algorithms import DQN, DiscreteActorCritic, DiscreteActorCriticParams, DQNParams

from .config import algorithm
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
    policy_creator = {f"{algorithm}_{i}.{i}": get_value_based_policy for i in range(4)}
    trainer_creator = {f"{algorithm}_{i}": get_dqn for i in range(4)}

elif algorithm == "ac":
    policy_creator = {f"{algorithm}_{i}.{i}": get_discrete_policy_gradient for i in range(4)}
    trainer_creator = {f"{algorithm}_{i}": get_ac for i in range(4)}
else:
    raise ValueError
