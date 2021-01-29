# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


def template(export_path: str, **kwargs):
    # Late import.
    from maro.cli.grass.executors.grass_executor import GrassExecutor

    GrassExecutor.template(
        export_path=export_path
    )
