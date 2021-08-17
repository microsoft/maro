# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum


class MsgTag(Enum):
    COLLECT = "rollout"
    EVAL = "eval"
    INIT_POLICIES = "init_policies"
    INIT_POLICIES_DONE = "init_policies_done"
    POLICY_STATE = "policy_state"
    CHOOSE_ACTION = "choose_action"
    ACTION = "action"
    LEARN = "learn"
    LEARN_DONE = "learn_finished"
    COMPUTE_GRAD = "compute_grad"
    COMPUTE_GRAD_DONE = "compute_grad_done"
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
    POLICY_NAMES = "policy_names"
    ROLLOUT_INFO = "rollout_info"
    TRACKER = "tracker"
    GRAD_TASK = "grad_task"
    LOSS_INFO = "loss_info"
    STATE = "state"
    POLICY_STATE = "policy_state"
    EXPLORATION_STEP = "exploration_step"
    VERSION = "version"
    NUM_STEPS = "num_steps"
    END_OF_EPISODE = "end_of_episode"
