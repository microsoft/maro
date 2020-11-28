from enum import Enum


class MessageTag(Enum):
    ROLLOUT = "rollout"
    UPDATE = "update"
    CHOOSE_ACTION = "choose_action"
    ACTION = "action"


class PayloadKey(Enum):
    ACTION = "action"
    AGENT_ID = "agent_id"
    MODEL = "model"
    EXPLORATION_PARAMS = "exploration_params"
    PERFORMANCE = "performance"
    DETAILS = "details"
    STATE = "state"
    SEED = "seed"
    DONE = "done"
    RETURN_DETAILS = "return_details"
