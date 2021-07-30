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
    LEARN = "LEARN"
    ABORT_ROLLOUT = "abort_rollout"
    EVAL_DONE = "eval_done"
    COLLECT_DONE = "collect_done"
    TRAIN_DONE = "train_done"
    DONE = "done"
    EXIT = "exit"


class MsgKey(Enum):
    ACTION = "action"
    AGENT_ID = "agent_id"
    EPISODE = "episode"
    SEGMENT = "segment"
    STEP = "step"
    EXPERIENCES = "experiences"
    TRACKER = "tracker"
    STATE = "state"
    POLICY_STATE = "policy_state"
    EXPLORATION_STEP = "exploration_step"
    VERSION = "version"
    NUM_STEPS = "num_steps"
    END_OF_EPISODE = "end_of_episode"
