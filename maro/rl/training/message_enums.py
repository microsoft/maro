# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum


class MsgTag(Enum):
    COLLECT = "rollout"
    EVAL = "eval"
    POLICY_UPDATE = "agent_update"
    CHOOSE_ACTION = "choose_action"
    ACTION = "action"
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
    ENV_SUMMARY = "env_summary"
    EXPERIENCES = "experiences"
    NUM_EXPERIENCES = "num_experiences"
    STATE = "state"
    POLICY = "policy"
    EXPLORATION = "exploration"
    VERSION = "version"
    NUM_STEPS = "num_steps"
    EPISODE_END = "episode_end"
