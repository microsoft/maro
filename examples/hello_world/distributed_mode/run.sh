#!/bin/sh

export PYTHONPATH=${PYTHONPATH}:/home/ysqyang/PycharmProjects/maro_release/

experiment=HelloWorld

# start components
GROUP=${experiment} INDEX=0 python3 learner.py > hw.txt &
GROUP=${experiment} INDEX=0 python3 environment_runner.py > hw.txt &