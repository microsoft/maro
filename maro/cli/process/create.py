# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import redis

from maro.cli.process.utils.default_param import process_setting
from maro.cli.process.utils.details import load_details, save_setting_info, start_agent, start_redis
from maro.cli.utils.params import LocalPaths, ProcessRedisName
from maro.utils.logger import CliLogger

logger = CliLogger(name=f"ProcessExecutor.{__name__}")


def create(deployment_path: str, **kwargs):
    current_process_path = os.path.expanduser(LocalPaths.MARO_PROCESS)
    # Create folder
    if not os.path.exists(current_process_path):
        os.makedirs(current_process_path)

    # Get environment setting
    setting_info = process_setting
    if deployment_path is not None:
        customized_setting = load_details(deployment_path=deployment_path)
        for key, value in customized_setting.items():
            if key in setting_info:
                setting_info[key] = value

    save_setting_info(setting_info)
    logger.info(f"MARO process mode setting: {setting_info}")

    # Start Redis
    if setting_info["redis_mode"] == "MARO":
        start_redis(port=setting_info["redis_info"]["port"])
        logger.info(f"Redis server start with port {setting_info['redis_info']['port']}.")

    redis_connection = redis.Redis(host=setting_info["redis_info"]["host"], port=setting_info["redis_info"]["port"])

    # Start agents
    start_agent()
    redis_connection.hset(ProcessRedisName.SETTING, "agent_status", 1)
    logger.info("Agents start.")

    # Push default setting into Redis
    del setting_info["redis_info"]
    redis_connection.hmset(ProcessRedisName.SETTING, setting_info)
