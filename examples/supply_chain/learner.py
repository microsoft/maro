# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from os.path import dirname, join, realpath
from maro.rl import DistributedLearner

sc_code_dir = dirname(realpath(__file__))

class SCLearner(DistributedLearner):
    def end_of_training(self, ep, **kwargs):
        pass
