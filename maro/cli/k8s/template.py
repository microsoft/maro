# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.cli.k8s.executors.k8s_azure_executor import K8sAzureExecutor


def template(export_path: str, **kwargs):
    K8sAzureExecutor.template(
        export_path=export_path
    )
