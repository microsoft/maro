#!/bin/bash

# script to start playground environment within docker container

./redis-6.0.6/src/redis-server -p 6379 &
redis-commander -p 40009 &

# Python 3.6
cd ./docs/_build/html; python -m http.server 40010 -b 0.0.0.0 &

# It's only for your play locally or in an internal network environment, so disable the token for convenience
cd ../../..; jupyter lab --port 40011 --allow-root --ip 0.0.0.0 --NotebookApp.token=''