#!/bin/bash

BASEDIR=$(dirname "$0")

# script to kill a previously launcher supply chain training job.
docker-compose -f $BASEDIR/../docker-compose.yml down
