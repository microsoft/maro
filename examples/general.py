# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
import yaml

example_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, example_dir)

from cim.env_wrapper import get_cim_env_wrapper
from cim.agent_wrapper import get_cim_agent_wrapper
from cim.meta import CIM_AGENT_IDS, CIM_CREATE_ROLLOUT_POLICY_FUNC, CIM_CREATE_TRAIN_POLICY_FUNC
from supply_chain.env_wrapper import SC_AGENT_IDS, get_sc_env_wrapper
from supply_chain.agent_wrapper import get_sc_agent_wrapper
from supply_chain.meta import SC_CREATE_ROLLOUT_POLICY_FUNC, SC_CREATE_TRAIN_POLICY_FUNC

config_path = os.path.join(example_dir, "config.yml")
with open(config_path, "r") as config_file:
    config = yaml.safe_load(config_file)

log_dir = os.path.join(example_dir, "logs", config["experiment_name"])

get_env_wrapper_func_index = {
    "cim": get_cim_env_wrapper,
    "sc": get_sc_env_wrapper    
} 
get_agent_wrapper_func_index = {
    "cim": get_cim_agent_wrapper,
    "sc": get_sc_agent_wrapper
}

agent_ids_index = {
    "cim": CIM_AGENT_IDS,
    "sc": SC_AGENT_IDS    
}

create_rollout_policy_func_index = {
    "cim": CIM_CREATE_ROLLOUT_POLICY_FUNC,
    "sc": SC_CREATE_ROLLOUT_POLICY_FUNC  
}

create_train_policy_func_index = {
    "cim": CIM_CREATE_TRAIN_POLICY_FUNC,
    "sc": SC_CREATE_TRAIN_POLICY_FUNC    
}
