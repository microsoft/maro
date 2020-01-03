#!/bin/bash
docker run --name $DOCKER_CONTAINER_NAME -d -it \
        --network host \
        -v $MOUNT_PATH:/maro_dev \
        -w $WORK_DIR \
        -e CONFIG=$CONFIG \
        $DOCKER_IMAGE_NAME $START_COMMAND

echo "<$DOCKER_CONTAINER_NAME> run <$START_COMMAND> in <$CONFIG> start!"