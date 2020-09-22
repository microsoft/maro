#!/bin/bash

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    cd "$(dirname $(readlink -f $0))/.."
elif [[ "$OSTYPE" == "darwin"* ]]; then
    cd "$(cd "$(dirname "$0")"; pwd -P)/.."
fi

bash ./scripts/build_maro.sh

# script to run all the test script under tests folder which the file name match test_xxxx.py

export PYTHONPATH="."

# install requirements
pip install -r ./tests/requirements.test.txt

coverage run --rcfile=./tests/.coveragerc

coverage report --rcfile=./tests/.coveragerc