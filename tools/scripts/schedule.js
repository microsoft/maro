// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.


// native lib
const fs = require('fs');
const path = require('path');
const cwd = process.cwd();
const os = require('os');

// 3rd party lib
const chalk = require('chalk');
const cliProgress = require('cli-progress');
const colors = require('colors');
const debug = require('debug')('schedule');
const inquirer = require('inquirer');
const yaml = require('yamljs');
const sendmail = require('sendmail')({
  silent: true
});

// private lib
const util = require(`${__dirname}/util`);
const ps = require(`${__dirname}/ps`);

const generateBatchSchedule = isTest => {
  const dateSuffix = new Date().toISOString().slice(0, 10).replace(/-|T|:/g, '_');
  const metaPath = path.join(cwd, 'schedule_meta');
  const schedule = {};
  const metaList = [];

  fs.readdirSync(metaPath).forEach(file => {
    if (file.includes('.yml')) {
      metaList.push(file);
    }
  });

  if (metaList.length == 0) {
    console.log(chalk.red('Create your schedule meta firstly!'));
    process.exit(1);
  }



  inquirer
    .prompt([
      {
        type: 'list',
        name: 'meta_file',
        choices: metaList,
        message: `What's your schedule meta file?`,
        default: 0
      }
    ])
    .then(input => {
      const meta = yaml.parse(fs.readFileSync(path.join(metaPath, input.meta_file), 'utf8'));
      const jobKeys = Object.keys(meta.jobs);
      const totalJobNum = jobKeys.length;
      let jobCounter = 0;
      const baseConfig = yaml.parse(fs.readFileSync(meta.base_config, 'utf8'));

      util.sexec(isTest, 'local', `git rev-parse --short HEAD`, (err, res) => {
        if (!err) {
          meta['commit'] = res.trim();
          util.sexec(isTest, 'local', `git branch | grep \\\* | cut -d ' ' -f2`, (err, res) => {
            meta['branch'] = res.trim();
            if (!err) {
              const schedulePath = path.join(cwd, 'schedule', dateSuffix, meta.name);
              let mkScheduleDir = `mkdir -p ${schedulePath}`;

              const finishedToken = ps.sub('config:finished', (err) => {
                if (!err) {
                  let jobSchedulePath = path.join(schedulePath, 'schedule.yml');
                  let yamlStr = yaml.dump(meta, { schema: 'DEFAULT_FULL_SCHEMA' });
                  fs.writeFileSync(jobSchedulePath, yamlStr, 'utf8');
                  ps.unSub('config:finished', finishedToken);
                  console.log(chalk.blue(jobSchedulePath));
                } else {
                  debug(chalk.red(err));
                }
              });

              util.sexec(isTest, 'local', mkScheduleDir, () => {
                for (let jobName of jobKeys) {
                  job = meta.jobs[jobName];
                  for (parameter of Object.keys(job.parameters)) {
                    let value = job.parameters[parameter];
                    let lastKey = baseConfig;
                    let keys = parameter.split('.');

                    for (let i = 0; i < keys.length; i++) {
                      if (i == keys.length - 1) {
                        lastKey[keys[i]] = value;
                        lastKey = baseConfig;
                        break;
                      }

                      if (lastKey[keys[i]] == undefined) {
                        lastKey[keys[i]] = {};
                      }

                      lastKey = lastKey[keys[i]];
                    }

                    jobConfigFile = path.join(schedulePath, `${jobName}.config.yml`);
                    configPathInDocker = path.join('/maro_dev', 'tools', 'schedule', dateSuffix, meta.name, `${jobName}.config.yml`);
                    job['state'] = 'pending';
                    job['config'] = jobConfigFile;
                    job['config_in_docker'] = configPathInDocker;
                  }
                  let yamlStr = yaml.dump(baseConfig, { schema: 'DEFAULT_FULL_SCHEMA' });
                  fs.writeFileSync(jobConfigFile, yamlStr, 'utf8');
                  jobCounter += 1;
                  if (jobCounter == totalJobNum) {
                    ps.pub('config:finished');
                  }
                }
              });
            }
          })
        }
      });
    })
    .catch(debug);
};

const runSchedule = (isTest) => {
  const scheduleDir = `${cwd}/schedule`;
  const scheduleList = [];
  fs.readdirSync(scheduleDir).forEach(date => {
    fs.readdirSync(`${scheduleDir}/${date}`).forEach(scheduleName => {
      scheduleList.unshift(path.join(scheduleDir, date, scheduleName, 'schedule.yml'));
    });
  });

  if (scheduleList.length == 0) {
    console.log(chalk.red('Generate your schedule firstly!'));
    process.exit(1);
  }

  inquirer.prompt([{
    type: 'list',
    name: 'schedule_file',
    message: `What's your schedule?`,
    choices: scheduleList,
    default: 0
  }])
    .then(input => {
      const schedule = yaml.parse(fs.readFileSync(input.schedule_file, 'utf8'));
      const runningJobDict = {};
      const finishedJobList = Object.keys(schedule.jobs)
        .filter(key => schedule.jobs[key].state == 'finished')
        .map(key => {
          let job = {
            key: key,
            payload: schedule.jobs[key]
          };

          return job;
        });
      const totalJobNum = Object.keys(schedule.jobs).length;
      const startTime = new Date().getTime();
      const pendingJobList = Object.keys(schedule.jobs)
        .filter(key => schedule.jobs[key].state == 'pending' || schedule.jobs[key].state == 'running')
        .map(key => {
          let job = {
            key: key,
            payload: schedule.jobs[key]
          };

          return job;
        });

      const splitScheduleLength = input.schedule_file.split('/').length;
      const scheduleFolder = input.schedule_file.split('/').slice(0, splitScheduleLength - 1).join('/');
      const scheduleBar = new cliProgress.SingleBar({
        format: 'Schedule Progress |' + colors.red('{bar}') + '| {percentage}% || ' +
          colors.yellow('finished: {value}/{total}, running: {running}/{max_parallel}, pending: {pending}/{total}') +
          ' || speed: {speed}s, eta: {eta}s, duration: {duration}s',
        barCompleteChar: '\u2588',
        barIncompleteChar: '\u2591',
        hideCursor: true
      });

      scheduleBar.start(totalJobNum, 0, {
        speed: 0,
        max_parallel: schedule.parallel_num,
        running: 0,
        pending: totalJobNum
      });

      const updateScheduleBar = () => {
        // update schedule progress bar data
        scheduleBar.update(finishedJobList.length, {
          speed: Math.ceil((new Date().getTime() - startTime) / 1000 / Math.min(finishedJobList.length + 1, totalJobNum)),
          running: Object.keys(runningJobDict).length,
          pending: pendingJobList.length,
          max_parallel: schedule.parallel_num
        });
      };

      updateScheduleBar();

      const cleanContainer = () => {
        for (let job of pendingJobList) {
          let rmContainer = `docker rm -f ${job.key}`;
          util.sexec(isTest, 'local', rmContainer);
        }

        for (let job of finishedJobList) {
          let rmContainer = `docker rm -f ${job.key}`;
          util.sexec(isTest, 'local', rmContainer);
        }
      };

      cleanContainer();

      const runSchedule = () => {
        // step1: check running job state
        let currentRunningJobNum = Object.keys(runningJobDict).length;
        let runningJobKeyList = Object.keys(runningJobDict);
        for (let i = 0; i < currentRunningJobNum; i++) {
          let runningJobKey = runningJobKeyList[i];
          let runningJob = runningJobDict[runningJobKey];
          let checkStateScript = `docker ps -a | grep ${runningJobKey} | egrep -o 'Exited \\([0-9]+\\)' | egrep -o '[0-9]+'`;
          util.sexec(isTest, 'local', checkStateScript, (err, res) => {
            if (!err) {
              let exitCode = parseInt(res.trim());
              runningJob.payload['exit_code'] = exitCode;
              runningJob.payload['state'] = 'finished';
              jobDuration = ((new Date().getTime() - runningJob.payload['duration']) / 1000 / 60).toFixed(2);
              runningJob.payload['duration'] = jobDuration + 'min';
              let tailLogFile = `${scheduleFolder}/${runningJobKey}.log`;
              let tailLog = `docker logs --tail 1000 ${runningJobKey} > ${tailLogFile}`;
              runningJob.payload['tail_log'] = tailLogFile;
              util.sexec(isTest, 'local', tailLog, () => {
                let rmFinishedContainer = `docker rm -f ${runningJobKey}`;
                util.sexec(isTest, 'local', rmFinishedContainer);
                finishedJobList.push(runningJob);
                delete runningJobDict[runningJobKey];
              });
            }
          });
        }

        // step2: lunch new job
        let availableNum = Math.min(
          schedule.parallel_num - Object.keys(runningJobDict).length,
          totalJobNum - finishedJobList.length
        );
        for (let i = 0; i < availableNum && pendingJobList.length > 0; i++) {
          pendingJob = pendingJobList.shift();
          pendingJob.payload['duration'] = new Date().getTime();
          pendingJob.payload['state'] = 'running';
          runningJobDict[pendingJob.key] = pendingJob;
          let runJobScript = [
            `DOCKER_CONTAINER_NAME=${pendingJob.key}`,
            `DOCKER_IMAGE_NAME=${schedule.docker_image}`,
            `CONFIG=${pendingJob.payload.config_in_docker}`,
            `START_COMMAND="${pendingJob.payload.run_cmd}"`,
            `MOUNT_PATH=${path.resolve(__dirname, '..', '..')}`,
            `WORK_DIR=${pendingJob.payload.work_dir}`,
            `bash ./scripts/run_job.sh`
          ].join(' ');
          pendingJob.payload['run_script'] = runJobScript;

          util.sexec(isTest, 'local', runJobScript);
          updateScheduleBar();
        }

        // step3: check schedule finished state
        if (finishedJobList.length == totalJobNum) {
          clearInterval(checkPendingJob);
          let yamlStr = yaml.dump(schedule, { schema: 'DEFAULT_FULL_SCHEMA' });
          fs.writeFileSync(input.schedule_file, yamlStr, 'utf8');
          if (schedule.auto_notification) {
            sendmail(
              {
                from: 'schedule@gmail.com',
                to: schedule.mail_to,
                subject: `${schedule.name} finished`,
                text: `${yaml.dump(schedule, {
                  schema: 'DEFAULT_FULL_SCHEMA'
                })}`
              },
              (err, res) => { }
            );
          }

          setTimeout(() => {
            updateScheduleBar();
            scheduleBar.stop();
            process.exit(0);
          }, 2000);
        }
      };

      process.on('SIGINT', () => {
        clearInterval(checkPendingJob);
        console.log(schedule);
        let yamlStr = yaml.dump(schedule, { schema: 'DEFAULT_FULL_SCHEMA' });
        fs.writeFileSync(input.schedule_file, yamlStr, 'utf8');
        process.exit(0);
      });

      process.on('uncaughtException', (err) => {
        clearInterval(checkPendingJob);
        console.log(schedule);
        let yamlStr = yaml.dump(schedule, { schema: 'DEFAULT_FULL_SCHEMA' });
        fs.writeFileSync(input.schedule_file, yamlStr, 'utf8');
        process.exit(1);
      });

      let checkPendingJob = setInterval(() => {
        runSchedule();
      }, 10000);
    })
    .catch(debug);
};

module.exports = {
  runSchedule: runSchedule,
  generateBatchSchedule: generateBatchSchedule
};
