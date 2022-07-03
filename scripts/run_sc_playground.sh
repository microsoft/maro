#!/bin/bash

container_name="sc_playground"
host_port="40010"

sc_topo_dir="maro/simulator/scenarios/supply_chain/topologies"

abs_local_dir=$(dirname $(cd "$(dirname "$0")"; pwd))

# docker run
docker run --name ${container_name} \
    -v ${abs_local_dir}/examples:/maro/examples \
    -v ${abs_local_dir}/notebooks:/maro/notebooks \
    -v ${abs_local_dir}/${sc_topo_dir}:/maro/${sc_topo_dir} \
    -p ${host_port}:40010 \
    -it maro_sc
