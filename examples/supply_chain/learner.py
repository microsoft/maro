# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl import Learner


class SCLearner(Learner):
    def end_of_training(self, ep, segment, **kwargs):
        pass
