# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


class PendingDecisionPayload:
    summary_key = ["ahu_idx"]

    def __init__(self, ahu_idx: int):
        self.ahu_idx = ahu_idx

    def __repr__(self):
        return "%s {ahu_idx: %r}" % (self.__class__.__name__, self.ahu_idx)


class Action:
    summary_key = ["ahu_idx", "sps", "das"]

    def __init__(self, ahu_idx: int, sps: float, das: float):
        self.ahu_idx = ahu_idx
        self.sps = sps
        self.das = das

    def __repr__(self):
        return "%s {ahu_idx: %r, sps: %r, das: %r}" % \
            (self.__class__.__name__, self.ahu_idx, self.sps, self.das)
