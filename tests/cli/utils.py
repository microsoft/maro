# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import time
from functools import wraps


def record_running_time(func_to_time: dict):
    def decorator(func):
        @wraps(func)
        def with_record_speed(*args, **kwargs):
            start_time = time.time_ns() / (10**9)
            func(*args, **kwargs)
            end_time = time.time_ns() / (10**9)
            func_to_time[func.__name__] = end_time - start_time

        return with_record_speed

    return decorator
