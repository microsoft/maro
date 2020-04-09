#!/bin/sh

projectpath=$PWD
mountpath="/maro"
net=maro
img=maro_dist
red=maro_redis

declare -A component

component[learner]=5
component[environment_runner]=2

cmdprefix="python3 /maro/examples/ecr/q_learning/distributed_mode/env_learner/"
rediscmd="redis-server"
experiment="t2"
logpath="/maro/examples/ecr/q_learning/distributed_mode/env_learner/${experiment}.html"

envopt="-e CONFIG=/maro/examples/ecr/q_learning/distributed_mode/env_learner/config.yml
        -e PYTHONPATH=/maro:/maro/maro/distributed/
        -e ENVNUM=${component[environment_runner]}
        -e LRNNUM=${component[learner]}
        -e LOGPATH=$logpath
        "

# start Redis server
docker network create $net

docker run -it -d --name $red                     \
           --network $net                         \
           --hostname $red                        \
           $img $rediscmd

echo "$red up and running"

# start components
for comp in "${!component[@]}"
do
  num=${component[$comp]}
  for (( i = 0; i < $num; i++ ))
  do
      cmd="${cmdprefix}${comp}.py -e ${experiment}"
      cntr=${comp}
      if [ $num -gt 1 ]
      then
        cmd="${cmd} -i ${i}"
        cntr="${cntr}_$i"
      fi
      docker run -it -d --name $cntr --network $net  \
                 -v $projectpath:$mountpath          \
                 ${envopt}                           \
                 $img $cmd
      echo "$cntr up and running"
  done
done