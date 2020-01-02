echo docker image: $DOCKER_IMAGE_NAME, docker container: $DOCKER_CONTAINER_NAME, mount path: $MOUNT_PATH

docker run --name $DOCKER_CONTAINER_NAME -d -it \
                --network host \
                -v $MOUNT_PATH:/maro_dev \
                $DOCKER_IMAGE_NAME