from maro.rl_v3.policy import DiscretePolicyGradient
from maro.rl_v3.training.algorithms import DiscreteMADDPG, DiscreteMADDPGParams
from maro.rl_v3.workflow import preprocess_get_policy_func_dict

from .config import algorithm, running_mode
from .nets import MyActorNet, MyMultiCriticNet


def get_multi_critic_net() -> MyMultiCriticNet:
    return MyMultiCriticNet()


ac_conf = DiscreteMADDPGParams(
    device="cpu",
    reward_discount=.0,
    num_epoch=10,
    get_q_critic_net_func=get_multi_critic_net,
    # shared_critic=True,
)


# #####################################################################################################################
def get_discrete_policy_gradient(name: str) -> DiscretePolicyGradient:
    return DiscretePolicyGradient(name=name, policy_net=MyActorNet())


def get_maddpg(name: str) -> DiscreteMADDPG:
    return DiscreteMADDPG(name=name, params=ac_conf)


if algorithm == "discrete_maddpg":
    get_policy_func = get_discrete_policy_gradient
    get_policy_func_dict = {
        f"{algorithm}_{i}.{i}": get_policy_func
        for i in range(4)
    }

    get_trainer_func_dict = {
        f"{algorithm}_{i}": get_maddpg
        for i in range(4)
    }
else:
    raise ValueError
# #####################################################################################################################

get_policy_func_dict = preprocess_get_policy_func_dict(
    get_policy_func_dict, running_mode
)
