# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

class Action:
    def __init__(self, assign: bool, vm_id: int=None, pm_id: int=None):
        self.assign = assign
        self.vm_id = vm_id
        self.pm_id = pm_id