#!/bin/bash

# script to build wheel package for linux
# NOTE: used in manylinux docker!

set -e -x

cd /maro

# Compile wheels
for PYVER in 6 7; do
    PYBIN="/opt/python/cp3${PYVER}-cp3${PYVER}m/bin"

    "${PYBIN}/pip" install -r maro/requirements.build.txt

    "${PYBIN}/python" setup.py bdist_wheel --plat-name manylinux2010_x86_64
done
