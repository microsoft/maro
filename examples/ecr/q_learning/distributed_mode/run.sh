#! /bin/sh

if [ -z "$CONFIG" ]
then
  config='config.yml'
else
  config=$CONFIG
fi

declare -A component
let component[learner]=$(yq r $config distributed.learner.num)
let component[environment_runner]=$(yq r $config distributed.environment_runner.num)

experiment=$(yq r $config experiment_name)

# start components
for comp in "${!component[@]}"
do
  num=${component[$comp]}
  for (( i = 0; i < $num; i++ ))
  do
      if [ $num -gt 1 ]
      then
          GROUP=${experiment} COMPTYPE=${comp} COMPID=${i} python3 ${comp}.py &
      else
          GROUP=${experiment} COMPTYPE=${comp} python3 ${comp}.py &
      fi
  done
done
