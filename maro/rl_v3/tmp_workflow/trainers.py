from maro.rl_v3.policy_trainer import DQN

algorithm = "dqn"
get_trainer_func_dict = {
    f"{algorithm}_trainer.{i}": lambda name: DQN(name=name) for i in range(4)
}

policy2trainer = {
    f"{algorithm}.{i}": f"{algorithm}_trainer.{i}" for i in range(4)
}
