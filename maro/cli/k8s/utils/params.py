# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os


class K8sPaths:
    MARO_K8S_LIB = "~/.maro/lib/k8s"
    ABS_MARO_K8S_LIB = os.path.expanduser(MARO_K8S_LIB)
