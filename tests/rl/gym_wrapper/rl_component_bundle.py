# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl.rl_component.rl_component_bundle import RLComponentBundle
from maro.simulator import Env

from tests.rl.gym_wrapper.config import algorithm, env_conf
from tests.rl.gym_wrapper.env_sampler import GymEnvSampler
from tests.rl.gym_wrapper.simulator.business_engine import GymBusinessEngine


env = Env(scenario="gym", business_engine_cls=GymBusinessEngine, **env_conf)
test_env = env

assert isinstance(env.business_engine, GymBusinessEngine)
gym_env = env.business_engine.gym_env
state_dim = gym_env.observation_space.shape[0]
action_dim = gym_env.action_space.shape[0]
is_continuous_act = True  # TODO
act_lower_bound, act_upper_bound = None, None
if is_continuous_act:
    act_lower_bound = gym_env.action_space.low
    act_upper_bound = gym_env.action_space.high

get_policy, get_trainer = None, None
if algorithm == "ac":
    from tests.rl.algorithms.ac import get_ac_policy, get_ac_trainer
    get_policy, get_trainer = get_ac_policy, get_ac_trainer
elif algorithm == "ppo":
    from tests.rl.algorithms.ppo import get_ppo_policy, get_ppo_trainer
    get_policy, get_trainer = get_ppo_policy, get_ppo_trainer
else:
    raise ValueError(f"Unsupported algorithm: {algorithm}")

name = f"{env_conf['topology']}_{algorithm}"
policy = get_policy(f"{name}.policy", state_dim, action_dim, is_continuous_act, act_lower_bound, act_upper_bound)
trainer = get_trainer(name, state_dim)
agent2policy = {0: f"{name}.policy"}

# Build RLComponentBundle
rl_component_bundle = RLComponentBundle(
    env_sampler=GymEnvSampler(
        learn_env=env,
        test_env=test_env,
        policies=[policy],
        agent2policy=agent2policy,
    ),
    agent2policy=agent2policy,
    policies=[policy],
    trainers=[trainer],
)
