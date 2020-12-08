from enum import Enum


class Component(Enum):
    LEARNER = "learner"
    ACTOR = "actor"


class MessageTag(Enum):
    ROLLOUT = "rollout"
    UPDATE = "update"
    CHOOSE_ACTION = "choose_action"
    ACTION = "action"
    EXIT = "exit"


class PayloadKey(Enum):
    ACTION = "action"
    ACTION_COUNT = "action_count"
    AGENT_ID = "agent_id"
    EPISODE = "episode"
    MODEL = "model"
    EXPLORATION_PARAMS = "exploration_params"
    PERFORMANCE = "performance"
    EXPERIENCES = "experiences"
    STATE = "state"
    RETURN_EXPERIENCES = "return_experiences"
