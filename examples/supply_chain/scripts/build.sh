#!/bin/bash

# script to build the docker image for running the supply chain scenario.
docker pull redis:6
docker build -f ../../../docker_files/dev.df -t maro-sc:latest ../../../
