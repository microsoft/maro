# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


class AHUSetPayload:
    summary_key = ["ahu_idx"]

    def __init__(self, ahu_idx: int):
        self.ahu_idx = ahu_idx
