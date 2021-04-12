#!/bin/bash

# script to run the supply chain scenario in single-host multi-container mode.
python3 docker_compose_yml_generator.py
docker-compose -f ../docker_compose_yamls/docker-compose.yml up
