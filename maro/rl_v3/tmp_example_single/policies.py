from maro.rl_v3.policy import DiscretePolicyGradient, ValueBasedPolicy
from maro.rl_v3.policy_trainer import DQN, DiscreteActorCritic
from maro.rl_v3.workflow import preprocess_get_policy_func_dict
from .config import ac_conf, algorithm, dqn_conf, dqn_policy_conf, running_mode
from .nets import MyActorNet, MyCriticNet, MyQNet

# ###############################################################################
if algorithm == "dqn":
    get_policy_func_dict = {
        f"{algorithm}.{i}": lambda name: ValueBasedPolicy(
            name=name, q_net=MyQNet(), device="cpu", **dqn_policy_conf) for i in range(4)
    }
    get_trainer_func_dict = {
        f"{algorithm}_trainer.{i}": lambda name: DQN(name=name, **dqn_conf) for i in range(4)
    }
elif algorithm == "ac":
    get_policy_func_dict = {
        f"{algorithm}.{i}": lambda name: DiscretePolicyGradient(
            name=name, policy_net=MyActorNet(), device="cpu") for i in range(4)
    }
    get_trainer_func_dict = {
        f"{algorithm}_trainer.{i}": lambda name: DiscreteActorCritic(
            name=name, get_v_critic_net_func=lambda: MyCriticNet(), **ac_conf
        ) for i in range(4)
    }
else:
    raise ValueError

get_policy_func_dict = preprocess_get_policy_func_dict(
    get_policy_func_dict, running_mode
)

policy2trainer = {
    f"{algorithm}.{i}": f"{algorithm}_trainer.{i}" for i in range(4)
}
