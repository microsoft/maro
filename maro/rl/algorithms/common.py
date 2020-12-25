# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import namedtuple

ActionWithLogProbability = namedtuple("action_with_probability", ["action", "log_probability"])
