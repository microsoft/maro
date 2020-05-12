# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from enum import Enum


class MsgType(Enum):
    STORE_EXPERIENCE = 0  # message contains actual experience data
    INITIAL_PARAMETERS = 1  # message contains model's parameter
    UPDATED_PARAMETERS = 2  # message notify the learner is ready for training
    NO_UPDATED_PARAMETERS = 3  # message to indicate that the learner has not collected enough experiences for training
    ENV_CHECKOUT = 4  # message notify the environment is finish and checkout


class MsgStatus(Enum):
    SEND_MESSAGE = 0
    RECEIVE_MESSAGE = 1
    

class PayloadKey(Enum):
    EXPERIENCE = 'experience'
    EPISODE = 'episode'
    POLICY_NET_PARAMETERS = 'policy_net_parameters'
    TARGET_NET_PARAMETERS = 'target_net_parameters'
    AGENT_ID = 'agent_id'  # agent's id
    AGENT_NAME = 'agent_name'  # agent's name
