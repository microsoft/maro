#!/bin/bash

# script to start playground environment within docker container

./redis-6.0.6/src/redis-server --port 6379 &
redis-commander --port 40009 &

# It's only for your play locally or in an internal network environment, so disable the token for convenience
jupyter lab --port 40010 --allow-root --ip 0.0.0.0 --NotebookApp.token=''
