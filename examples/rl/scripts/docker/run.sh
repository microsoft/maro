#!/bin/bash

NAMESPACE=${1:-maro}
BASEDIR=$(dirname "$0")

# script to run the multi-container mode.
python3 $BASEDIR/docker_compose_yml.py --namespace $NAMESPACE
docker-compose -f $BASEDIR/yq.yml --project-name $NAMESPACE up