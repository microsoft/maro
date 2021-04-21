from enum import Enum


class MsgTag(Enum):
    ROLLOUT = "rollout"
    AGENT_UPDATE = "agent_update"
    CHOOSE_ACTION = "choose_action"
    ACTION = "action"
    EXPERIENCE_SYNC = "experience_sync"
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
    EXPERIENCES = "experiences"
    NUM_EXPERIENCES = "num_experiences"
    STATE = "state"
    TRAINING = "training"
    POLICY = "policy"
    VERSION = "version"
    EXPLORATION_PARAMS = "exploration_params"
    NUM_STEPS = "num_steps"
    SEGMENT_INDEX = "segment_index"
    RETURN_ENV_METRICS = "return_env_metrics"
    TOTAL_REWARD = "total_reward"
    ENV_END = "env_end"
