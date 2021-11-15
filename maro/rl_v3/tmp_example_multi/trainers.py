from maro.rl_v3.policy_trainer import DiscreteMultiActorCritic
from .config import ac_conf, algorithm
from .policies import MyMultiCriticNet

if algorithm == "maac":
    get_trainer_func_dict = {
        f"{algorithm}.{i}_trainer": lambda name: DiscreteMultiActorCritic(
            name=name, get_v_critic_net_func=lambda: MyMultiCriticNet(), device="cpu", **ac_conf
        ) for i in range(4)
    }
else:
    raise ValueError

policy2trainer = {
    f"{algorithm}.{i}": f"{algorithm}.{i}_trainer" for i in range(4)
}
