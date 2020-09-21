#!/bin/bash

# script to build maro locally on linux/mac, usually for development
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    cd "$(dirname $(readlink -f $0))/.."
elif [[ "$OSTYPE" == "darwin"* ]]; then
    cd "$(cd "$(dirname "$0")"; pwd -P)/.."
fi

# compile cython files first
bash ./scripts/compile_cython.sh

python setup.py build_ext -i