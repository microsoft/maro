#! /bin/sh

if [ -z "$CONFIG" ]
then
  config='config.yml'
else
  config=$CONFIG
fi

declare -A component
let component[learner]=$(yq r $config distributed.learner.num)
let component[actor]=$(yq r $config distributed.actor.num)

experiment=$(yq r $config experiment_name)

# start components
for comp in "${!component[@]}"
do
  num=${component[$comp]}
  for (( i = 0; i < $num; i++ ))
  do
    GROUP=${experiment} COMPTYPE=${comp} python ${comp}.py &
  done
done
