import torch

from maro.rl.exploration import epsilon_greedy, MultiLinearExplorationScheduler
from maro.rl_v3.policy import DiscretePolicyGradient, ValueBasedPolicy
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
dqn_conf = {
    "reward_discount": .0,
    "update_target_every": 5,
    "num_epochs": 10,
    "soft_update_coef": 0.1,
    "double": False,
    "replay_memory_capacity": 10000,
    "random_overwrite": False,
    "train_batch_size": 32
}
ac_conf = {
    "reward_discount": .0,
    "grad_iters": 10,
    "critic_loss_cls": torch.nn.SmoothL1Loss,
    "min_logp": None,
    "lam": .0
}

# #####################################################################################################################
if algorithm == "dqn":
    train_param = {"device": "cpu", **dqn_conf}
    trainer_param_dict = {
        f"{algorithm}_{i}": train_param
        for i in range(4)
    }

    get_policy_func = lambda name: ValueBasedPolicy(name=name, q_net=MyQNet(), **dqn_policy_conf)
    get_policy_func_dict = {
        f"{algorithm}_{i}.{i}": get_policy_func
        for i in range(4)
    }
elif algorithm == "ac":
    train_param = {"device": "cpu", "get_v_critic_net_func": lambda: MyCriticNet(), **ac_conf}
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
