from enum import Enum


class MsgTag(Enum):
    ROLLOUT = "rollout"
    CHOOSE_ACTION = "choose_action"
    ACTION = "action"
    TRAJECTORY_SYNC = "experience_sync"
    TRAIN = "train"
    ABORT_ROLLOUT = "abort_rollout"
    ROLLOUT_DONE = "rollout_done"
    EXIT = "exit"


class MsgKey(Enum):
    ACTION = "action"
    AGENT_ID = "agent_id"
    ROLLOUT_INDEX = "rollout_index"
    TIME_STEP = "time_step"
    METRICS = "metrics"
    TRAJECTORY = "trajectory"
    EXPERIENCE = "experience"
    STATE = "state"
    TRAINING = "training"
    MODEL = "model"
    EXPLORATION_PARAMS = "exploration_params"
