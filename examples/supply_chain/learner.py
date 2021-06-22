# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from os.path import dirname, join, realpath
from maro.rl import DistributedLearner
from render_tools import SimulationTracker

sc_code_dir = dirname(realpath(__file__))

class SCLearner(DistributedLearner):
    
    def _evaluate(self, ep: int):
        tracker = SimulationTracker(60, 1, self.eval_env, self)
        loc_path = '/maro/supply_chain/output/'
        facility_types = ["productstore"]
        tracker.run_and_render(loc_path, facility_types)
    
    def end_of_training(self, ep, **kwargs):
        pass
