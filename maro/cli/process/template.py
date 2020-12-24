# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import shutil

from maro.cli.utils.params import LocalPaths


def template(setting_deploy, export_path, **kwargs):
    deploy_files = os.listdir(LocalPaths.MARO_PROCESS_DEPLOYMENT)
    if not setting_deploy:
        deploy_files.remove("process_setting_deployment.yml")
    export_path = os.path.abspath(export_path)
    for file_name in deploy_files:
        if os.path.isfile(f"{LocalPaths.MARO_PROCESS_DEPLOYMENT}/{file_name}"):
            shutil.copy(f"{LocalPaths.MARO_PROCESS_DEPLOYMENT}/{file_name}", export_path)
