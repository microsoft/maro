from enum import Enum


class PayloadKey(Enum):
    MODEL = "model"
    EXPLORATION_PARAMS = "exploration_params"
    PERFORMANCE = "performance"
    DETAILS = "details"
    SEED = "seed"
    DONE = "done"
    RETURN_DETAILS = "return_details"
