from enum import Enum


class Component(Enum):
    LEARNER = "learner"
    ACTOR = "actor"


class MessageTag(Enum):
    ROLLOUT = "rollout"
    ACTION = "action"
    FINISH = "finish"
    RESET = "reset"
    EXIT = "exit"


class PayloadKey(Enum):
    ACTION = "action"
    AGENT_ID = "agent_id"
    EPISODE = "episode"
    MODEL = "model"
    EXPLORATION_PARAMS = "exploration_params"
    PERFORMANCE = "performance"
    EXPERIENCES = "experiences"
    STATE = "state"
    IS_TRAINING = "is_training"
