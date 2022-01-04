from maro.rl_v3.policy import DiscretePolicyGradient
from maro.rl_v3.workflow import preprocess_get_policy_func_dict

from .config import algorithm, running_mode
from .nets import MyActorNet, MyMultiCriticNet
from ..training.algorithms import DistributedDiscreteMADDPG
from ..training.algorithms.maddpg import MADDPGParams

ac_conf = MADDPGParams(
    device="cpu",
    reward_discount=.0,
    num_epoch=10,
    get_q_critic_net_func=lambda: MyMultiCriticNet(),
    # shared_critic=True,
)

# #####################################################################################################################
if algorithm == "discretemaddpg":
    get_policy_func = lambda name: DiscretePolicyGradient(name=name, policy_net=MyActorNet())
    get_policy_func_dict = {
        f"{algorithm}_{i}.{i}": get_policy_func
        for i in range(4)
    }

    get_trainer_func_dict = {
        f"{algorithm}_{i}": lambda name: DistributedDiscreteMADDPG(name=name, params=ac_conf)
        for i in range(4)
    }
else:
    raise ValueError
# #####################################################################################################################

get_policy_func_dict = preprocess_get_policy_func_dict(
    get_policy_func_dict, running_mode
)
