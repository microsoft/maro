# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.cli.process.executor import ProcessExecutor


def delete(**kwargs):
    executor = ProcessExecutor()
    executor.delete()
