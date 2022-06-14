# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script is used to launch data and vis services, and the start the experiment script.

It accept a path to experiment launch file:


python launch_realtime_vis.py D:\projects\python\maro\examples\hello_world\cim\hello.py


maro vis service start/stop

maro start path/exp


steps:

1. launch the servcies' docker-compose.yml if services not started.

2. lauch the experiment file


"""
