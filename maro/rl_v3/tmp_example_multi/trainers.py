from config import ac_conf, algorithm
from maro.rl_v3.policy_trainer import DiscreteMultiActorCritic
from policies import MyMultiCriticNet

if algorithm == "maac":
    get_trainer_func_dict = {
        f"{algorithm}_trainer": lambda name: DiscreteMultiActorCritic(
            name=name, get_v_critic_net_func=lambda: MyMultiCriticNet(), **ac_conf
        )
    }
else:
    raise ValueError

policy2trainer = {
    f"{algorithm}.{i}": f"{algorithm}_trainer" for i in range(4)
}
