// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Set default env.
process.env['DEBUG'] = process.env['DEBUG'] || '*';

// 3rd party lib
const argv = require('yargs').argv;
const gulp = require('gulp');

// private lib
const util = require(`${__dirname}/scripts/util`);
const docker = require(`${__dirname}/scripts/docker`);
const schedule = require(`${__dirname}/scripts/schedule`);
const dashboard = require(`${__dirname}/scripts/dashboard`);
const ecr_config = require(`${__dirname}/scripts/ecr_config`);

// options
const isTest = argv.test == undefined ? false : true;
const isSync = argv.sync == undefined ? false : true;
const isForce = argv.force == undefined ? false : true;

/*--------------------------help task start-----------------------------*/
gulp.task('default', () => {
  util.help(gulp);
});

// /**
//  * Remote debug.
//  * @task {l/debug}
//  * @arg {test} Test flag, not run cmd, only output cmd.
//  */
// gulp.task('l/debug', () => {
//   util.remoteDebug(isTest);
// });

/**
 * Write docs.
 * @task {l/write_docs}
 * @arg {test} Test flag, not run cmd, only output cmd.
 */
gulp.task('l/write_docs', () => {
  util.writeDocs(isTest);
});

/**
 * Host docs.
 * @task {l/host_docs}
 * @arg {test} Test flag, not run cmd, only output cmd.
 * @arg {sync} Sync mode, which will auto refresh browser.
 */
gulp.task('l/host_docs', () => {
  util.hostDocs(isTest, isSync);
});

/**
 * Build docs.
 * @task {l/build_docs}
 * @arg {test} Test flag, not run cmd, only output cmd.
 */
gulp.task('l/build_docs', () => {
  util.buildDocs(isTest);
});

/**
 * Convert markdown to rst.
 * @task {l/md2rst}
 * @arg {test} Test flag, not run cmd, only output cmd.
 */
gulp.task('l/md2rst', () => {
  util.md2Rst(isTest);
});

/**
 * Build maro simulator.
 * @task {l/build_sim}
 * @arg {test} Test flag, not run cmd, only output cmd.
 */
gulp.task('l/build_sim', () => {
  util.buildSim(isTest);
});
/*--------------------------help task end-----------------------------*/

/*--------------------------docker task end-----------------------------*/
/**
 * Local build docker image.
 * @task {l/build_image}
 * @arg {test} Test flag, not run cmd, only output cmd.
 */
gulp.task('l/build_image', () => {
  docker.buildImage(isTest);
});

/**
 * Local launch docker container.
 * @task {l/launch_container}
 * @arg {test} Test flag, not run cmd, only output cmd.
 */
gulp.task('l/launch_container', () => {
  docker.launchContainer(isTest);
});

/**
 * Local login docker container.
 * @task {l/login_container}
 * @arg {test} Test flag, not run cmd, only output cmd.
 */
gulp.task('l/login_container', () => {
  docker.loginContainer(isTest);
});

/**
 * Clean exited container.
 * @task {l/clean_container}
 * @arg {test} Test flag, not run cmd, only output cmd.
 * @arg {force} Force flag, will rm all container, which includes exit and up container.
 */
gulp.task('l/clean_container', () => {
  docker.cleanContainer(isTest, isForce);
});

/**
 * Clean exited image.
 * @task {l/clean_image}
 * @arg {test} Test flag, not run cmd, only output cmd.
 */
gulp.task('l/clean_image', () => {
  docker.cleanImage(isTest, isForce);
});

/**
 * List exited image.
 * @task {l/list_image}
 * @arg {test} Test flag, not run cmd, only output cmd.
 */
gulp.task('l/list_image', () => {
  docker.listImage(isTest, isForce);
});

/**
 * List exited container.
 * @task {l/list_container}
 * @arg {test} Test flag, not run cmd, only output cmd.
 */
gulp.task('l/list_container', () => {
  docker.listContainer(isTest, isForce);
});
/*--------------------------docker task end-----------------------------*/

/*--------------------------schedule task start-----------------------------*/
/**
 * Generate batch schedule.
 * @task {s/generate_batch_schedule}
 * @arg {test} Test flag, not run cmd, only output cmd.
 */
gulp.task('s/generate_batch_schedule', () => {
  schedule.generateBatchSchedule(isTest);
});

/**
 * Run schedule.
 * @task {s/run}
 * @arg {test} Test flag, not run cmd, only output cmd.
 */
gulp.task('s/run', () => {
  schedule.runSchedule(isTest);
});
/*--------------------------schedule task end-----------------------------*/

/*--------------------------dashboard task end-----------------------------*/
/**
 * Local build docker image for dashboard.
 * @task {d/build_image}
 * @arg {test} Test flag, not run cmd, only output cmd.
 */
gulp.task('d/build_image', () => {
  dashboard.buildImage(isTest);
});

/**
 * Local start docker container for dashboard.
 * @task {d/startService}
 * @arg {test} Test flag, not run cmd, only output cmd.
 */
gulp.task('d/startService', () => {
  dashboard.startService(isTest);
});

/**
 * Local stop docker container for dashboard.
 * @task {d/stopService}
 * @arg {test} Test flag, not run cmd, only output cmd.
 */
gulp.task('d/stopService', () => {
  dashboard.stopService(isTest);
});

/*--------------------------docker task end-----------------------------*/
/*--------------------------ecr task start-----------------------------*/
/**
 * Generate ECR configs.
 * @task {ecr/gen_config}
 * @arg {test} Test flag, not run cmd, only output cmd.
 */
gulp.task('ecr/gen_config', () => {
  ecr_config.generateConfig(isTest);
});

/**
 * Visualize order distribution based on ECR configs.
 * @task {ecr/draw_order}
 * @arg {test} Test flag, not run cmd, only output cmd.
 */
gulp.task('ecr/draw_order', () => {
  ecr_config.visualizeOrder(isTest);
});

/**
 * Visualize ECR topologies.
 * @task {ecr/draw_topo}
 * @arg {test} Test flag, not run cmd, only output cmd.
 */
gulp.task('ecr/draw_topo', () => {
  ecr_config.visualizeTopology(isTest);
});
/*--------------------------ecr task end-----------------------------*/
