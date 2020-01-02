# How to run distributed mode example

## Local run

### [Prerequisites](../../../../README.md)

### How to install yq if need
[Install yq](http://mikefarah.github.io/yq/)

### Run Examples

```sh
cd env_learner
# set config file path
export CONFIG=./config.yml
# start redis
bash start_redis.sh
# start distributed mode
bash run.sh
```
