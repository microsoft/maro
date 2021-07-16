#!/bin/bash

# Script to install maro in editable mode on linux/darwin,
# usually for development.

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    cd "$(dirname $(readlink -f $0))/.."
elif [[ "$OSTYPE" == "darwin"* ]]; then
    cd "$(cd "$(dirname "$0")"; pwd -P)/.."
fi

# Install dependencies.
pip install -r ./maro/requirements.build.txt

# Compile cython files.
bash scripts/compile_cython.sh

# Install MARO in editable mode.
pip install -e .
