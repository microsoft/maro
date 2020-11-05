# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.utils.utils import check_deployment_status, deploy

from .__misc__ import __data_version__, __version__

if not check_deployment_status():
    deploy()
