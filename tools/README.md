# How to run

## Pre-Request

### [node.js (>=8.0)](https://nodejs.org/en/download/)

```shell
curl -sL https://deb.nodesource.com/setup_8.x | sudo -E bash - && apt-get install -y nodejs
```

### Gulp (3.9.1)

```shell
npm install --global gulp-cli
npm install --save gulp@3.9.1
```

### 3rd Party Packages

```shell
npm install
```

### [docker](https://docs.docker.com/v17.09/engine/installation/)

## Run

```shell
gulp
```

### Batch run experiments
#### Pre-Request
Create/deploy your own schedule meta in `./schedule_meta`.

#### How to use it?
```sh
# generate schedule in silent mode
DEBUG=schedule gulp s/generate_batch_schedule
# generate schedule in verbose mode
gulp s/generate_batch_schedule

# run schedule in silent mode
sudo DEBUG=schedule gulp s/run
# run schedule in verbose mode
sudo gulp s/run
```
