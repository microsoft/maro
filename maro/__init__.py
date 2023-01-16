# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# isort: skip_file

from .__misc__ import __data_version__, __version__

from maro.utils.utils import check_deployment_status, deploy

if not check_deployment_status():
    deploy()
