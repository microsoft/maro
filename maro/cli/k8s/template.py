# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


def template(export_path: str, **kwargs):
    # Late import.
    from maro.cli.k8s.executors.k8s_aks_executor import K8sAksExecutor

    K8sAksExecutor.template(
        export_path=export_path
    )
