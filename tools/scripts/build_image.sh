#! /bin/bash
echo sample: "DOCKER_FILE=../../docker_files/cpu.no_mount.rls.df DOCKER_FILE_DIR=../.. DOCKER_IMAGE_NAME=maro/rls/cpu bash ./build_image.sh"
echo docker file: $DOCKER_FILE, docker file dir: $DOCKER_FILE_DIR, docker image name: $DOCKER_IMAGE_NAME

docker build -f $DOCKER_FILE $DOCKER_FILE_DIR -t $DOCKER_IMAGE_NAME