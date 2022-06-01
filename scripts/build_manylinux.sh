#!/bin/bash

# this script used build maro packages for linux

cd "$(dirname $(readlink -f $0))/.."

bash ./scripts/compile_cython.sh

# NOTE: currently we only support python3.6 and 3.7, need to be clearfy the python and packages version
# about manylinux: https://github.com/pypa/manylinux
docker run --rm -v "$PWD":/maro quay.io/pypa/manylinux2010_x86_64 bash /maro/scripts/build_wheel.sh
