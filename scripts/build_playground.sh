#!/bin/bash

# script to build docker for playground image on linux/mac, this require the source code of maro

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    cd "$(dirname $(readlink -f $0))/.."
elif [[ "$OSTYPE" == "darwin"* ]]; then
    cd "$(cd "$(dirname "$0")"; pwd -P)/.."
fi

bash ./scripts/compile_cython.sh

docker build -f ./docker_files/cpu.playground.df . -t maro2020/playground
