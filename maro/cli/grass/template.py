# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.cli.grass.executors.grass_executor import GrassExecutor


def template(export_path: str, **kwargs):
    GrassExecutor.template(
        export_path=export_path
    )
