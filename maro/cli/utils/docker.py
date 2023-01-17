# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import docker


def image_exists(image_name: str):
    try:
        client = docker.from_env()
        client.images.get(image_name)
        return True
    except docker.errors.ImageNotFound:
        return False


def build_image(context: str, docker_file_path: str, image_name: str):
    client = docker.from_env()
    with open(docker_file_path, "r"):
        client.images.build(
            path=context,
            tag=image_name,
            quiet=False,
            rm=True,
            custom_context=False,
            dockerfile=docker_file_path,
        )


def push(local_image_name: str, repository: str):
    client = docker.from_env()
    image = client.images.get(local_image_name)
    acr_tag = f"{repository}/{local_image_name}"
    image.tag(acr_tag)
    # subprocess.run(f"docker push {acr_tag}".split())
    client.images.push(acr_tag)
    print(f"Pushed image to {acr_tag}")
