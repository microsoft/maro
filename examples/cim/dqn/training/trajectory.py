# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict

from maro.rl import AbsActor

from examples.cim.common import CIMTrajectory


class CIMTrajectoryForDQN(CIMTrajectory):
    def on_finish(self):
        exp_by_agent = defaultdict(lambda: defaultdict(list))
        for i in range(len(self.trajectory["state"]) - 1):
            agent_id = list(state.keys())[0]
            exp = exp_by_agent[agent_id]
            exp["S"].append(self.trajectory["state"][i][agent_id])
            exp["A"].append(self.trajectory["action"][i][agent_id])
            exp["R"].append(self.get_offline_reward(self.trajectory["event"][i]))
            exp["S_"].append(list(self.trajectory["state"][i + 1].values())[0])

        return dict(exp_by_agent)
