#!/bin/bash

BASEDIR=$(dirname "$0")
echo $BASEDIR

# script to kill a previously launched training job.
docker-compose -f $BASEDIR/docker-compose.yml down
rm $BASEDIR/docker-compose.yml