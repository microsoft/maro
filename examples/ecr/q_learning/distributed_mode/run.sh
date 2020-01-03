#! /bin/sh

declare -A component
let component[learner]=$(yq r $CONFIG distributed.learner.num)
let component[environment_runner]=$(yq r $CONFIG distributed.environment_runner.num)

experiment=$(yq r $CONFIG experiment_name)

# start components
for comp in "${!component[@]}"
do
  num=${component[$comp]}
  for (( i = 0; i < $num; i++ ))
  do
      if [ $num -gt 1 ]
      then
          GROUP=${experiment} INDEX=${i} python3 ${comp}.py &
      else
          GROUP=${experiment} python3 ${comp}.py &
      fi
  done
done
