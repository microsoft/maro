# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from os.path import dirname, join, realpath
from maro.rl import Learner

sc_code_dir = dirname(realpath(__file__))

class SCLearner(Learner):
    def end_of_training(self, ep, segment, **kwargs):
        pass
