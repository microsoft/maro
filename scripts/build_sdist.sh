#!/bin/bash

# script to create source package on linux

cd "$(dirname $(readlink -f $0))/.."

bash ./scripts/compile_cython.sh

python setup.py sdist