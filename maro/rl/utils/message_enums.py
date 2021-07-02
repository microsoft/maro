# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum


class MsgTag(Enum):
    COLLECT = "rollout"
    EVAL = "eval"
    INIT_POLICY_STATE = "init_policy_state"
    INIT_POLICY_STATE_DONE = "init_policy_state_done"
    GET_INITIAL_POLICY_STATE = "get_initial_policy_state"
    POLICY_STATE = "policy_state"
    CHOOSE_ACTION = "choose_action"
    ACTION = "action"
    TRAIN = "train"
    ABORT_ROLLOUT = "abort_rollout"
    EVAL_DONE = "eval_done"
    COLLECT_DONE = "collect_done"
    DONE = "done"
    EXIT = "exit"


class MsgKey(Enum):
    ACTION = "action"
    AGENT_ID = "agent_id"
    EPISODE = "episode"
    SEGMENT = "segment"
    STEP = "step"
    ENV_SUMMARY = "env_summary"
    EXPERIENCES = "experiences"
    NUM_EXPERIENCES = "num_experiences"
    STATE = "state"
    POLICY_STATE = "policy_state"
    EXPLORATION_STEP = "exploration_step"
    VERSION = "version"
    NUM_STEPS = "num_steps"
    EPISODE_END = "episode_end"
