#!/bin/bash

BASEDIR=$(dirname "$0")

# script to run the CIM scenario in single-host multi-container mode.
python3 $BASEDIR/docker_compose_yml_generator.py
docker-compose -f $BASEDIR/../docker-compose.yml up