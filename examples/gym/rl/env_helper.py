from maro.simulator import Env
from maro.simulator.scenarios.gym.business_engine import GymBusinessEngine

env_conf = {
    "scenario": "gym",
    "start_tick": 0,
    "durations": 5000,
    "options": {
        "random_seed": None,
    },
}

helper_env = Env(**env_conf)
be = helper_env.business_engine
assert isinstance(be, GymBusinessEngine)

gym_env = be.gym_env
gym_state_dim = gym_env.observation_space.shape[0]
gym_action_dim = gym_env.action_space.shape[0]
# action_lower_bound = gym_env.action_space.low.tolist()
# action_upper_bound = gym_env.action_space.high.tolist()
action_lower_bound = [float('-inf') for _ in range(gym_env.action_space.shape[0])]  # TODO
action_upper_bound = [float('inf') for _ in range(gym_env.action_space.shape[0])]  # TODO
