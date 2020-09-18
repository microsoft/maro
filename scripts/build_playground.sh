#!/bin/bash

# script to build docker for playground image on linux/mac, this require the source code of maro

cd "$(dirname $(readlink -f $0))/.."

bash ./scripts/compile_cython.sh

docker build -f ./docker_files/cpu.play.df . -t maro/playground:cpu