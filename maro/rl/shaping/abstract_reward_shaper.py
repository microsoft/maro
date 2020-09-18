# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict, deque
from abc import ABC, abstractmethod

from maro.rl.common import ExperienceKey, TransitionInfoKey


class AbstractRewardShaper(ABC):
    """
    A reward shaper is used to record transitions during a roll-out episode and perform necessary post-processing   \n
    at the end of the episode. The post-processing logic is encapsulated in the abstract shape() method and needs   \n
    to be implemented for each scenario. It is necessary to compute rewards and next-states (and also next-actions  \n
    for SARSA-like on-policy algorithms) during post-processing as they are set to None during the episode. In      \n
    particular, it is necessary to specify how to determine the reward for an action given the business metrics     \n
    associated with the corresponding transition. MARO provides the KStepRewardShaper class which  may be combined  \n
    with a user-defined reward function to form a default reward shaper.
    """
    def __init__(self, **kwargs):
        self._trajectory = defaultdict(deque)

    def push(self, transition_dict: dict):
        for key, trans in transition_dict.items():
            self._trajectory[key].append(trans)

    def __call__(self, snapshot_list):
        self._shape(snapshot_list)
        return self._pack()

    def _pack(self):
        """
        Retrieves experiences for individual agents from the trajectory data after shaping.
        """
        exp_by_agent = {}
        while len(self._trajectory[ExperienceKey.STATE]) > 1:
            agent_id = self._trajectory["extra"].popleft()[TransitionInfoKey.AGENT_ID]
            if agent_id not in exp_by_agent:
                exp_by_agent[agent_id] = defaultdict(list)
            for key in ExperienceKey:
                exp_by_agent[agent_id][key].append(self._trajectory[key].popleft())
            exp_by_agent[agent_id]["info"].append(self._trajectory["info"].popleft())

        return exp_by_agent

    @abstractmethod
    def _shape(self, snapshot_list):
        return NotImplementedError
