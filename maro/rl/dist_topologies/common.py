from enum import Enum


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
