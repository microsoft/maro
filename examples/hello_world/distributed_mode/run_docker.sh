#!/bin/sh

projectpath="/home/ysqyang/PycharmProjects/maro_release"
mountpath="/maro"
net=maro
img=maro_dist
red=maro_redis

env=environment_runner
lrn=learner

cmdprefix="python3 /maro/examples/hello_world/distributed_mode/"
rediscmd="redis-server"

# start Redis server
docker network create $net

docker run -it -d --name $red                     \
           --network $net                         \
           --hostname $red                        \
           $img $rediscmd

echo "$red up and running"

# start components
for comp in ${env} ${lrn}
do
  cmd="${cmdprefix}${comp}.py"
  docker run -it -d --name $comp --network $net  \
             -v $projectpath:$mountpath          \
             -e PYTHONPATH=/maro                 \
             $img $cmd
  echo "$comp up and running"
done
