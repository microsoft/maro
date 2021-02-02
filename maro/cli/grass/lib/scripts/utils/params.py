# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os


class Paths:
    MARO_SHARED = "~/.maro-shared"
    ABS_MARO_SHARED = os.path.expanduser(MARO_SHARED)

    MARO_LOCAL = "~/.maro-local"
    ABS_MARO_LOCAL = os.path.expanduser(MARO_LOCAL)
