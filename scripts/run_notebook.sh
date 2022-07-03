#!/bin/bash

nohup jupyter lab --port 40010 --allow-root --ip 0.0.0.0 --NotebookApp.token='' > jupyterlab.log 2>&1 &

/bin/bash
