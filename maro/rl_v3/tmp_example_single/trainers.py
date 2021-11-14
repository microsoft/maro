from maro.rl_v3.policy_trainer import DQN, DiscreteActorCritic
from .config import ac_conf, algorithm, dqn_conf
from .policies import MyCriticNet

if algorithm == "dqn":
    get_trainer_func_dict = {
        f"{algorithm}_trainer.{i}": lambda name: DQN(name=name, **dqn_conf) for i in range(4)
    }
elif algorithm == "ac":
    get_trainer_func_dict = {
        f"{algorithm}_trainer.{i}": lambda name: DiscreteActorCritic(
            name=name, get_v_critic_net_func=lambda: MyCriticNet(), **ac_conf
        ) for i in range(4)
    }
else:
    raise ValueError

policy2trainer = {
    f"{algorithm}.{i}": f"{algorithm}_trainer.{i}" for i in range(4)
}
