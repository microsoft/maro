from abc import abstractmethod
from typing import Dict, List

from maro.rl_v3.policy_learner import AbsLearner


class AbsLearnerManager(object):
    def __init__(self) -> None:
        super(AbsLearnerManager, self).__init__()

    @abstractmethod
    def learn(self) -> None:
        pass

    @abstractmethod
    def get_policy_states(self) -> Dict[str, Dict[str, object]]:
        pass


class SimpleLearnerManager(AbsLearnerManager):
    def __init__(
        self,
        learners: List[AbsLearner]
    ) -> None:
        super(SimpleLearnerManager, self).__init__()
        self._learners = learners

    def learn(self) -> None:
        for learner in self._learners:
            learner.train_step()

    def get_policy_states(self) -> Dict[str, Dict[str, object]]:
        return {learner.name: learner.get_policy_state_dict() for learner in self._learners}
