from maro.simulator import Env
from maro.simulator.scenarios.gym.business_engine import GymBusinessEngine

if __name__ == "__main__":
    env = Env(
        scenario="gym",
        start_tick=0,
        durations=10,
    )
    be = env.business_engine
    assert isinstance(be, GymBusinessEngine)
    gym_env = be.gym_env

    print(gym_env.action_space.low)
    print(gym_env.action_space.high)

    metrics, decision_event, is_done = env.step(None)
    while not is_done:
        action = gym_env.action_space.sample()
        print(f"State at tick {env.tick}: {decision_event}")
        print(f"Action: {action}")
        metrics, decision_event, is_done = env.step(action)
