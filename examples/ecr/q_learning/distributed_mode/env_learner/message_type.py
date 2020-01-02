# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from enum import Enum


class MsgType(Enum):
    STORE_EXPERIENCE = 0  # message contains actual experience data
    INITIAL_PARAMETERS = 1  # message contains model's parameter
    UPDATED_PARAMETERS = 2  # message notify the learner is ready for training
    ENV_CHECKOUT = 3  # message notify the environment is finish and checkout


class MsgKey(Enum):
    EXPERIENCE = 'experience'
    EPISODE = 'episode'
    POLICY_NET_PARAMETERS = 'policy_net_parameters'
    TARGET_NET_PARAMETERS = 'target_net_parameters'
    AGENT_ID = 'agent_id'  # agent's id
    AGENT_NAME = 'agent_name'  # agent's name
