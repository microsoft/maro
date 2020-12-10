# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum


class Component(Enum):
    ACTOR = "actor"
    TRAINER = "trainer"


class MessageTag(Enum):
    EXPLORATION_PARAMS = "exploration_params"
    EXPLORATION_PARAMS_ACK = "exploration_params_ack"
    UPDATE = "update"
    CHOOSE_ACTION = "choose_action"
    ACTION = "action"
    MODEL = "model"
    TRAINING_FINISHED = "training_finished"
    EXIT = "exit"


class PayloadKey(Enum):
    ACTION = "action"
    AGENT_ID = "agent_id"
    MODEL = "model"
    EXPLORATION_PARAMS = "exploration_params"
    EXPERIENCES = "experiences"
    STATE = "state"
