# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import yaml

from maro.cli.process.executor import ProcessExecutor
from maro.cli.process.utils.default_param import process_setting


def create(deployment_path: str, **kwargs):
    if deployment_path is not None:
        with open(deployment_path, "r") as fr:
            create_deployment = yaml.safe_load(fr)
    else:
        create_deployment = process_setting

    executor = ProcessExecutor(create_deployment)
    executor.create()
