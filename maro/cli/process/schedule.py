# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.cli.process.executor import ProcessExecutor


def start_schedule(deployment_path: str, **kwargs):
    executor = ProcessExecutor()
    executor.start_schedule(deployment_path=deployment_path)


def stop_schedule(schedule_name: str, **kwargs):
    executor = ProcessExecutor()
    executor.stop_schedule(schedule_name=schedule_name)
