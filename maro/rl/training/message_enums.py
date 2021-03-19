from enum import Enum


class MessageTag(Enum):
    ROLLOUT = "rollout"
    CHOOSE_ACTION = "choose_action"
    ACTION = "action"
    EXPERIENCE = "experience_sync"
    ABORT_ROLLOUT = "abort_rollout"
    TRAIN = "train"
    FINISHED = "finished"
    EXIT = "exit"


class PayloadKey(Enum):
    ACTION = "action"
    AGENT_ID = "agent_id"
    ROLLOUT_INDEX = "rollout_index"
    TIME_STEP = "time_step"
    METRICS = "metrics"
    EXPERIENCE = "experience"
    STATE = "state"
    TRAINING = "training"
    MODEL = "model"
    EXPLORATION_PARAMS = "exploration_params"
