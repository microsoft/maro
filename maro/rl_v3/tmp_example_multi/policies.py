from maro.rl_v3.policy import DiscretePolicyGradient
from maro.rl_v3.workflow import preprocess_get_policy_func_dict

from .config import algorithm, running_mode
from .nets import MyActorNet, MyMultiCriticNet

ac_conf = {
    "reward_discount": .0,
    "num_epoch": 10
}

# #####################################################################################################################
if algorithm == "discretemaddpg":
    train_param = {"device": "cpu", "get_q_critic_net_func": lambda: MyMultiCriticNet(), **ac_conf}
    trainer_param_dict = {
        f"{algorithm}_{i}": train_param
        for i in range(4)
    }

    get_policy_func = lambda name: DiscretePolicyGradient(name=name, policy_net=MyActorNet())
    get_policy_func_dict = {
        f"{algorithm}_{i}.{i}": get_policy_func
        for i in range(4)
    }
else:
    raise ValueError
# #####################################################################################################################

get_policy_func_dict = preprocess_get_policy_func_dict(
    get_policy_func_dict, running_mode
)
