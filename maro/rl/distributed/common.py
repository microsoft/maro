from enum import Enum


class LearnerActorComponent(Enum):
    LEARNER = "learner"
    ACTOR = "actor"


class ActorTrainerComponent(Enum):
    ACTOR = "actor"
    TRAINER = "trainer"


class MessageTag(Enum):
    ROLLOUT = "rollout"
    EXPLORATION_PARAMS = "exploration_params"
    UPDATE = "update"
    CHOOSE_ACTION = "choose_action"
    ACTION = "action"
    MODEL = "model"
    EXIT = "exit"


class PayloadKey(Enum):
    ACTION = "action"
    AGENT_ID = "agent_id"
    MODEL = "model"
    EXPLORATION_PARAMS = "exploration_params"
    PERFORMANCE = "performance"
    EXPERIENCES = "experiences"
    STATE = "state"
    SEED = "seed"
    DONE = "done"
    RETURN_DETAILS = "return_experiences"
