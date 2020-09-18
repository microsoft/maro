# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys


def exit(state: int = 0, msg: str = None):
    """Exit and show msg in sys.stderr"""
    if msg is not None:
        sys.stderr.write(msg)

    sys.exit(state)
