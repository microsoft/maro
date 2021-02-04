from enum import Enum


class MessageTag(Enum):
    ROLLOUT = "rollout"
    CHOOSE_ACTION = "choose_action"
    ACTION = "action"
    TERMINATE_EPISODE = "terminate_episode"
    FINISHED = "finished"
    EXIT = "exit"


class PayloadKey(Enum):
    ACTION = "action"
    AGENT_ID = "agent_id"
    EPISODE = "episode"
    TIME_STEP = "time_step"
    MODEL = "model"
    EXPLORATION_PARAMS = "exploration_params"
    PERFORMANCE = "performance"
    DETAILS = "details"
    STATE = "state"
    IS_TRAINING = "is_training"
    ROLLOUT_KWARGS = "rollout_kwargs"
    ACTOR_CLIENT_ID = "actor_client_id"


# type definition for the special action that should be used to terminate a roll-out episode
class TerminateRollout:
    pass
