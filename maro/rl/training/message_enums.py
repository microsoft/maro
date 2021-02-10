from enum import Enum


class MessageTag(Enum):
    ROLLOUT = "rollout"
    CHOOSE_ACTION = "choose_action"
    ACTION = "action"
    ABORT_ROLLOUT = "abort_rollout"
    FINISHED = "finished"
    EXIT = "exit"


class PayloadKey(Enum):
    ACTION = "action"
    AGENT_ID = "agent_id"
    ROLLOUT_INDEX = "rollout_index"
    TIME_STEP = "time_step"
    METRICS = "metrics"
    DETAILS = "details"
    STATE = "state"
    TRAINING = "training"
    ROLLOUT_KWARGS = "rollout_kwargs"
    ACTOR_CLIENT_ID = "actor_client_id"
