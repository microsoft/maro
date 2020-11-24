from enum import Enum


class PayloadKey(Enum):
    MODEL = "model"
    EPSILON = "epsilon"
    PERFORMANCE = "performance"
    DETAILS = "details"
    SEED = "seed"
    DONE = "done"
    RETURN_DETAILS = "return_details"
