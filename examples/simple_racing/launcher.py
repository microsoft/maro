# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

from maro.simulator import Env

from maro.simulator.scenarios.simple_racing.business_engine import SimpleRacingBusinessEngine
from maro.simulator.scenarios.simple_racing.common import Action, DecisionEvent


def choose_action(state: np.ndarray) -> np.ndarray:
    # TODO: Implement the logic to get the action here.
    action: np.ndarray = None
    return action


if __name__ == "__main__":
    env = Env(scenario="simple_racing", topology="example", durations=1000)
    assert isinstance(env._business_engine, SimpleRacingBusinessEngine)

    metrics, decision_payload, is_done = env.step(None)

    while not is_done:
        assert isinstance(decision_payload, DecisionEvent)
        # TODO: Update the usage of DecisionEvent and Creation of Action here if you update the class definition.
        action = Action(choose_action(decision_payload.state))
        metrics, decision_payload, is_done = env.step(action)

    env.reset()
