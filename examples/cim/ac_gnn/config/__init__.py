# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from datetime import datetime

from .agent_config import agent_config
from .training_config import training_config

time_str = datetime.now().strftime("%m/%d/%Y_%H:%M:%S")
training_config["group"] = f"{training_config['group']}_{time_str}"

__all__ = ["agent_config", "training_config"]
