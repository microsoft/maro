from enum import Enum


class PayloadKey(Enum):
    RolloutMode = "rollout_mode"
    MODEL = "model"
    EPSILON = "epsilon"
    PERFORMANCE = "performance"
    EXPERIENCE = "experience"
    SEED = "seed"
