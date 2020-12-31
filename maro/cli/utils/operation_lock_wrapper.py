# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os
import re

import yaml

from maro.cli.utils.params import GlobalPaths
from maro.utils.exception.cli_exception import BadRequestError
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)

ALLOW_PATTERNS = {
    # No locks at create

    # clean related
    "clean": {r"^.*$"},

    # data related
    "push_data": {r"^.*data$", r"^.*image$"},
    "pull_data": {r"^.*data$", r"^.*image$"},
    "remove_data": {r"^.*data$", r"^.*image$"},

    # delete related
    "delete": {r"$^"},

    # image related
    "push_image": {r"^.*data$", r"^list.*$", r"^.*node$"},

    # job related
    "start_job": {r"^list.*$"},
    "stop_job": {r"^list.*$"},

    # node related
    "scale_node": {r"^list.*$", r"^.*data$"},
    "start_node": {r"^list.*$", r"^.*data$"},
    "stop_node": {r"^list.*$", r"^.*data$"},
    "list_node": {r"^.*$"},

    # schedule related
    "start_schedule": {r"^list.*$"},
    "stop_schedule": {r"^list.*$"},
}


def operation_lock(func):
    def with_lock(*args, **kwargs):
        _acquire_lock(
            cluster_name=kwargs["cluster_name"], operation=func.__name__
        )
        try:
            func(*args, **kwargs)
        finally:
            _release_lock(
                cluster_name=kwargs["cluster_name"], operation=func.__name__
            )

    return with_lock


def _save_lock(cluster_name: str, details: dict) -> None:
    with open(os.path.expanduser(f"{GlobalPaths.MARO_CLUSTERS}/{cluster_name}/.lock"), "w") as fw:
        yaml.safe_dump(details, fw)


def _load_lock(cluster_name: str) -> dict:
    if not os.path.exists(os.path.expanduser(f"{GlobalPaths.MARO_CLUSTERS}/{cluster_name}/.lock")):
        return {}
    with open(os.path.expanduser(f"{GlobalPaths.MARO_CLUSTERS}/{cluster_name}/.lock"), "r") as fr:
        details = yaml.safe_load(fr)
    return details


def _acquire_lock(cluster_name: str, operation: str):
    logger.debug(f"Acquire lock at {cluster_name} with operation {operation}")

    # Load details
    lock_details = _load_lock(cluster_name=cluster_name)
    logger.debug(f"Before acquire: {lock_details}")

    # Check locks
    for lock_operation in lock_details:
        allow_acquire = False
        for pattern in ALLOW_PATTERNS[lock_operation]:
            if bool(re.match(pattern, operation)):
                allow_acquire = True
                break
        if not allow_acquire:
            raise BadRequestError(f"Unable to execute command, a running operation: '{lock_operation}' blocks it.")

    # Save details
    lock_details[operation] = lock_details.get(operation, 0) + 1
    logger.debug(f"After acquire: {lock_details}")
    _save_lock(cluster_name, lock_details)


def _release_lock(cluster_name: str, operation: str):
    logger.debug(f"Release lock at {cluster_name} with operation {operation}")

    # Delete operation
    if operation == "delete":
        logger.debug("Skip lock release")
        return

    # Load details
    lock_details = _load_lock(cluster_name=cluster_name)
    logger.debug(f"Before release: {lock_details}")

    lock_details[operation] -= 1
    if lock_details[operation] == 0:
        lock_details.pop(operation)

    # Save details
    logger.debug(f"After release: {lock_details}")
    _save_lock(cluster_name, lock_details)
