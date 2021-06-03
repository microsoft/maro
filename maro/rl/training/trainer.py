# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from collections import defaultdict
from os import getcwd
from typing import Dict, List

from maro.communication import Proxy, SessionType
from maro.rl.experience import ExperienceSet
from maro.rl.policy import AbsPolicy, AbsCorePolicy
from maro.utils import Logger

from .message_enums import MsgKey, MsgTag


class Trainer:
    def __init__(self) -> None:
        pass

    def run(self):
        pass
