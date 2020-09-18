#!/bin/bash

# script to build maro locally on linux/mac, usually for development

cd "$(dirname $(readlink -f $0))/.."

# compile cython files first
bash ./scripts/compile_cython.sh

python setup.py build_ext -i