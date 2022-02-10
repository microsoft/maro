# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib
import os
import pickle
import socket
import sys
from types import ModuleType
from typing import List, Union


def from_env(var_name: str, required: bool = True, default: object = None) -> object:
    if var_name not in os.environ:
        if required:
            raise KeyError(f"Missing environment variable: {var_name}")
        else:
            return default

    var = os.getenv(var_name)
    if var.isnumeric() or var[0] == "-" and var[1:].isnumeric():
        return int(var)

    try:
        return float(var)
    except ValueError:
        return var


def from_env_as_int(var_name: str, required: bool = True, default: object = None) -> int:
    ret = from_env(var_name, required, default)
    assert isinstance(ret, int)
    return ret


def from_env_as_float(var_name: str, required: bool = True, default: object = None) -> float:
    ret = from_env(var_name, required, default)
    assert isinstance(ret, float)
    return ret


def get_eval_schedule(sch: Union[int, List[int]], num_episodes: int) -> List[int]:
    """Helper function to the policy evaluation schedule.

    Args:
        sch (Union[int, List[int]]): Evaluation schedule. If it is an int, it is treated as the number of episodes
            between two adjacent evaluations. For example, if the total number of episodes is 20 and ``sch`` is 5,
            this will return [5, 10, 15, 20]. If it is a list, it will return a sorted version of the list.
        num_episodes (int): Total number of learning episodes.

    Returns:
        A list of episodes indicating when to perform policy evaluation.

    """
    if sch is None:
        schedule = []
    elif isinstance(sch, int):
        num_eval_schedule = num_episodes // sch
        schedule = [sch * i for i in range(1, num_eval_schedule + 1)]
    else:
        schedule = sorted(sch)

    return schedule


# serialization and deserialization for messaging
DEFAULT_MSG_ENCODING = "utf-8"


def string_to_bytes(s: str) -> bytes:
    return s.encode(DEFAULT_MSG_ENCODING)


def bytes_to_string(bytes_: bytes) -> str:
    return bytes_.decode(DEFAULT_MSG_ENCODING)


def pyobj_to_bytes(pyobj) -> bytes:
    return pickle.dumps(pyobj)


def bytes_to_pyobj(bytes_: bytes) -> object:
    return pickle.loads(bytes_)


def get_own_ip_address() -> str:
    return socket.gethostbyname(socket.gethostname())


def get_ip_address_by_hostname(host: str) -> str:
    while True:
        try:
            return socket.gethostbyname(host)
        except Exception:
            continue
