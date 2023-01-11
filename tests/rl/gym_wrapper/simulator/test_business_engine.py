from maro.simulator import Env

from tests.rl.gym_wrapper.simulator.business_engine import GymBusinessEngine
from tests.rl.gym_wrapper.simulator.common import Action

if __name__ == "__main__":
    env = Env(
        scenario="gym",
        topology="Walker2d-v4",
        start_tick=0,
        durations=10,
        business_engine_cls=GymBusinessEngine,
    )
    be = env.business_engine
    assert isinstance(be, GymBusinessEngine)
    gym_env = be.gym_env

    print(gym_env.action_space.low)
    print(gym_env.action_space.high)

    metrics, decision_event, is_done = env.step(None)
    while not is_done:
        gym_action = gym_env.action_space.sample()
        print(f"State at tick {env.tick}: {decision_event.state}")
        print(f"Action: {gym_action}")
        metrics, decision_event, is_done = env.step(Action(gym_action))
