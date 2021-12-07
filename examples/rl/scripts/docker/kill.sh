#!/bin/bash

NAMESPACE=${1:-maro}
BASEDIR=$(dirname "$0")

# script to kill a previously launched training job.
docker-compose -f $BASEDIR/yq.yml --project-name $NAMESPACE down

rm $BASEDIR/yq.yml