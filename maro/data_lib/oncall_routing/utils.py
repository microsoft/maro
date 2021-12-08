# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

def convert_time_format(t: int) -> int:
    return (t // 100) * 60 + (t % 100)
