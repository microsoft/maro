![MARO LOGO](../../../../docs/source/images/logo.svg)
# How to run distributed mode example

Before running the distributed examples, please meet the following prerequisites.
### [Prerequisites](../../../../README.md)

### How to install yq if need
[Install yq](https://mikefarah.gitbook.io/yq/)

### Run Distributed Examples

```sh
# set config file path
export CONFIG=./config.yml
# start redis
bash start_redis.sh
# start distributed mode
bash run.sh
```

## License

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the [MIT](./LICENSE) License.