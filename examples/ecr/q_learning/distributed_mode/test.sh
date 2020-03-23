#!/bin/sh

projectpath=$(dirname $(dirname $(dirname $(dirname "$PWD"))))
mountpath="/maro_dist"
net=maro
img=dist:latest
red=maro_redis

workpath="/maro_dist/examples/ecr/q_learning/distributed_mode/"
rediscmd="redis-server"

#generate job component config
python3 gen_job_component_config.py
wait

echo "job component config generated"

# start Redis server
docker network create $net

# docker run -p 6379:6379 -d  \
#            --name $red      \
#            --network $net   \
#            $img /bin/zsh $rediscmd   \

# docker run -it -d --name $red   \
#            --network $net       \
#            --hostname $red      \
#            $img $rediscmd

# echo "$red up and running"

#start component
for config in `ls ./job_component_config/`
do
    echo $config
    cntid=$(yq r ./job_component_config/$config self_id)
    component_type=$(yq r ./job_component_config/$config self_component_type)
    experiment_name=$(yq r ./job_component_config/$config experiment_name)
    envopt="-e CONFIG=${workpath}/job_component_config/${config}" 
    cmd="python3 ${workpath}$component_type.py -e $experiment_name"
    echo $img
    docker run -it -d --name $cntid         \
               --network host               \
               -v $projectpath:$mountpath   \
               ${envopt}                    \
               $img $cmd
    echo "$cntid up and running"
done