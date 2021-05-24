# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum


class MsgTag(Enum):
    COLLECT = "rollout"
    EVAL = "eval"
    AGENT_UPDATE = "agent_update"
    CHOOSE_ACTION = "choose_action"
    ACTION = "action"
    EXPERIENCE_SYNC = "experience_sync"
    TRAIN = "train"
    ABORT_ROLLOUT = "abort_rollout"
    EVAL_DONE = "eval_done"
    COLLECT_DONE = "collect_done"
    EXIT = "exit"


class MsgKey(Enum):
    ACTION = "action"
    AGENT_ID = "agent_id"
    EPISODE_INDEX = "episode_index"
    SEGMENT_INDEX = "segment_index"
    TIME_STEP = "time_step"
    METRICS = "metrics"
    EXPERIENCES = "experiences"
    NUM_EXPERIENCES = "num_experiences"
    STATE = "state"
    POLICY = "policy"
    VERSION = "version"
    NUM_STEPS = "num_steps"
    RETURN_ENV_METRICS = "return_env_metrics"
    TOTAL_REWARD = "total_reward"
    EPISODE_END = "episode_end"
