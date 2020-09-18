#!/bin/bash

cd "$(dirname $(readlink -f $0))/.."

bash ./scripts/build_maro.sh

# script to run all the test script under tests folder which the file name match test_xxxx.py

export PYTHONPATH="."

# install requirements
pip install -r ./tests/requirements.txt

coverage run --rcfile=./tests/.coveragerc

coverage report --rcfile=./tests/.coveragerc