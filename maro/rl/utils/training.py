# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os


def get_latest_ep(path: str) -> int:
    ep_list = [int(ep) for ep in os.listdir(path)]
    return max(ep_list)
