# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

from maro.cli.utils.config_parser import format_env_vars, get_script_path, get_mnt_path_in_container


def get_job_manifest(agent_pool_name: str, component_name: str, container_spec: dict, volumes: List[dict]):
    return {
        "metadata": {"name": component_name},
        "spec": {
            "template": {
                "spec": {
                    "nodeSelector": {"agentpool": agent_pool_name},
                    "restartPolicy": "Never",
                    "volumes": volumes,
                    "containers": [container_spec]
                }
            }
        }
    }


def get_azurefile_volume_spec(name: str, share_name: str, secret_name: str):
    return {
        "name": name,
        "azureFile": {
            "secretName": secret_name,
            "shareName": share_name,
            "readOnly": False
        }
    }


def get_container_spec(image_name: str, component_name: str, redis_host: str, redis_port: int, env: dict, volumes):
    common_container_spec = {
        "image": image_name,
        "imagePullPolicy": "Always",
        "volumeMounts": [{"name": vol["name"], "mountPath": get_mnt_path_in_container(vol["name"])} for vol in volumes]
    }
    return {
        **common_container_spec,
        **{
            "name": component_name,
            "command": ["python3", get_script_path(component_name, containerized=True)],
            "env": format_env_vars({**env, "REDIS_HOST": redis_host, "REDIS_PORT": str(redis_port)}, mode="k8s")
        }
    }


def get_redis_deployment_manifest(host: str, port: int):
    return {
        "metadata": {
            "name": host,
            "labels": {"app": "redis"}
        },
        "spec": {
            "selector": {
                "matchLabels": {"app": "redis"}
            },
            "replicas": 1,
            "template": {
                "metadata": {
                    "labels": {"app": "redis"}
                },
                "spec": {
                    "containers": [
                        {
                            "name": "master",
                            "image": "redis:6",
                            "ports": [{"containerPort": port}]
                        }
                    ]
                }
            }
        }
    }


def get_redis_service_manifest(host: str, port: int):
    return {
        "metadata": {
            "name": host,
            "labels": {"app": "redis"}
        },
        "spec": {
            "ports": [{"port": port, "targetPort": port}],
            "selector": {"app": "redis"}
        }
    }


def get_cross_namespace_service_access_manifest(
    service_name: str, target_service_name: str, target_service_namespace: str, target_service_port: int
):
    return {
        "metadata": {
            "name": service_name,
        },
        "spec": {
            "type": "ExternalName",
            "externalName": f"{target_service_name}.{target_service_namespace}.svc.cluster.local",
            "ports": [{"port": target_service_port}]
        }
    }
