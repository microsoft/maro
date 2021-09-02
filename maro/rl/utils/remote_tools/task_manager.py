# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import namedtuple

LearnTask = namedtuple("LearnTask", ["policy_name", "model_state", "batch"])


class TaskManager:
    pass
