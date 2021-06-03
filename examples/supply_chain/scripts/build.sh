#!/bin/bash

BASEDIR=$(dirname "$0")
ROOTDIR=$BASEDIR/../../../

# script to build the docker image for running the supply chain scenario.
docker pull redis:6
docker build -f $ROOTDIR/docker_files/dev.df -t maro-sc:latest $ROOTDIR
