@ECHO off

@REM define the variables
set container_name=sc_playground
set host_port=40010

@REM the dir of sc topologies
set sc_topo_dir=maro/simulator/scenarios/supply_chain/topologies

@REM get the root dir of maro
set abs_local_dir=%~dp0\..\

@REM docker run
docker run --name %container_name% ^
		-v %abs_local_dir%/examples:/maro/examples ^
    -v %abs_local_dir%/notebooks:/maro/notebooks ^
    -v %abs_local_dir%/%sc_topo_dir%:/maro/%sc_topo_dir% ^
    -p %host_port%:40010 ^
    -it maro2020/maro_sc
