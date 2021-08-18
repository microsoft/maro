#!/bin/bash

BASEDIR=$(dirname "$0")

# script to run the multi-container mode.
python3 $BASEDIR/docker_compose_yml.py
docker-compose -f $BASEDIR/docker-compose.yml up