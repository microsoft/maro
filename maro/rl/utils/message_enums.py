# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum


class MsgTag(Enum):
    SAMPLE = "sample"
    TEST = "test"
    SAMPLE_DONE = "eval_done"
    TEST_DONE = "collect_done"
    INIT_POLICIES = "init_policies"
    INIT_POLICIES_DONE = "init_policies_done"
    POLICY_STATE = "policy_state"
    CHOOSE_ACTION = "choose_action"
    ACTION = "action"
    GET_INITIAL_POLICY_STATE = "get_initial_policy_state"
    LEARN = "learn"
    LEARN_DONE = "learn_finished"
    COMPUTE_GRAD = "compute_grad"
    COMPUTE_GRAD_DONE = "compute_grad_done"
    ABORT_ROLLOUT = "abort_rollout"
    DONE = "done"
    EXIT = "exit"
    REQUEST_WORKER = "request_worker"
    RELEASE_WORKER = "release_worker"
    ASSIGN_WORKER = "assign_worker"


class MsgKey(Enum):
    ACTION = "action"
    AGENT_ID = "agent_id"
    EPISODE = "episode"
    SEGMENT = "segment"
    NUM_STEPS = "num_steps"
    STEP = "step"
    POLICY_IDS = "policy_ids"
    ROLLOUT_INFO = "rollout_info"
    INTO = "info"
    GRAD_TASK = "grad_task"
    GRAD_SCOPE = "grad_scope"
    LOSS_INFO = "loss_info"
    STATE = "state"
    TENSOR = "tensor"
    POLICY_STATE = "policy_state"
    EXPLORATION_STEP = "exploration_step"
    VERSION = "version"
    STEP_RANGE = "step_range"
    END_OF_EPISODE = "end_of_episode"
    WORKER_ID = "worker_id"
    WORKER_ID_LIST = "worker_id_list"
