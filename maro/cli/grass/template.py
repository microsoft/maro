# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.cli.grass.executors.grass_azure_executor import GrassAzureExecutor


def template(export_path: str, **kwargs):
    GrassAzureExecutor.template(
        export_path=export_path
    )
