# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.cli.process.executor import ProcessExecutor


def start_schedule(deployment_path: str):
    executor = ProcessExecutor(deployment_path=deployment_path)
    executor.start_schedule()


def stop_schedule(schedule_name: str):
    executor = ProcessExecutor(name=schedule_name)
    executor.stop_schedule()
