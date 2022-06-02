# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import kubernetes
from kubernetes import client, config


def load_config():
    config.load_kube_config()


def create_namespace(namespace: str):
    body = client.V1Namespace()
    body.metadata = client.V1ObjectMeta(name=namespace)
    try:
        client.CoreV1Api().create_namespace(body)
    except kubernetes.client.exceptions.ApiException:
        pass


def create_deployment(conf: dict, namespace: str):
    client.AppsV1Api().create_namespaced_deployment(namespace, conf)


def create_service(conf: dict, namespace: str):
    client.CoreV1Api().create_namespaced_service(namespace, conf)


def create_job(conf: dict, namespace: str):
    client.BatchV1Api().create_namespaced_job(namespace, conf)


def create_secret(name: str, data: dict, namespace: str):
    client.CoreV1Api().create_namespaced_secret(
        body=client.V1Secret(metadata=client.V1ObjectMeta(name=name), data=data),
        namespace=namespace,
    )


def delete_job(namespace: str):
    client.BatchV1Api().delete_collection_namespaced_job(namespace)
    client.CoreV1Api().delete_namespace(namespace)


def describe_job(namespace: str):
    client.CoreV1Api().read_namespace(namespace)
