# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .__misc__ import __version__, __data_version__

from maro.utils.utils import deploy, check_deployment_status

if not check_deployment_status():
    deploy()
