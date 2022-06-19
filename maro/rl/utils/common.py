# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pickle
import socket
from typing import Any, List, Optional


def get_env(var_name: str, required: bool = True, default: str = None) -> Optional[str]:
    """Wrapper for os.getenv() that includes a check for mandatory environment variables.

    Args:
        var_name (str): Variable name.
        required (bool, default=True): Flag indicating whether the environment variable in questions is required.
            If this is true and the environment variable is not present in ``os.environ``, a ``KeyError`` is raised.
        default (str, default=None): Default value for the environment variable if it is missing in ``os.environ``
            and ``required`` is false. Ignored if ``required`` is True.

    Returns:
        The environment variable.
    """
    if var_name not in os.environ:
        if required:
            raise KeyError(f"Missing environment variable: {var_name}")
        return default

    return os.getenv(var_name)


def int_or_none(val: Optional[str]) -> Optional[int]:
    return int(val) if val is not None else None


def float_or_none(val: Optional[str]) -> Optional[float]:
    return float(val) if val is not None else None


def list_or_none(vals_str: Optional[str]) -> List[int]:
    return [int(val) for val in vals_str.split()] if vals_str is not None else []


# serialization and deserialization for messaging
DEFAULT_MSG_ENCODING = "utf-8"


def string_to_bytes(s: str) -> bytes:
    return s.encode(DEFAULT_MSG_ENCODING)


def bytes_to_string(bytes_: bytes) -> str:
    return bytes_.decode(DEFAULT_MSG_ENCODING)


def pyobj_to_bytes(pyobj: Any) -> bytes:
    return pickle.dumps(pyobj)


def bytes_to_pyobj(bytes_: bytes) -> Any:
    return pickle.loads(bytes_)


def get_own_ip_address() -> str:
    """https://stackoverflow.com/questions/166506/finding-local-ip-addresses-using-pythons-stdlib"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0)
    try:
        # doesn't even have to be reachable
        sock.connect(("10.255.255.255", 1))
        ip = sock.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        sock.close()
    return ip


def get_ip_address_by_hostname(host: str) -> str:
    if host in ("localhost", "127.0.0.1"):
        return get_own_ip_address()

    while True:
        try:
            return socket.gethostbyname(host)
        except Exception:
            continue
