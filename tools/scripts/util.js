// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.


// native lib
const os = require('os');
const path = require('path');

// 3rd party lib
const art = require('ascii-art');
const chalk = require('chalk');
const chalkAni = require('chalk-animation');
const dateFormat = require('dateformat');
const debug = require('debug')('util');
const gulp = require('gulp');
const watch = require('gulp-watch');
const inquirer = require('inquirer');
const ora = require('ora');
const pexec = require('child-process-promise').exec;
const publicIp = require('public-ip');
const usage = require('gulp-help-doc');

// private lib
const version = '0.1.0';

const chalks = {
  local: {
    value: '[L]',
    chalk: chalk.white.bgBlue
  },
  remote: {
    value: '[R]',
    chalk: chalk.white.bgMagenta
  }
};

const tagger = env => {
  let tags = env.split('|');
  let res = '';
  for (tag of tags) {
    if (chalks[tag] != undefined) {
      res += `${chalks[tag].chalk(`${chalks[tag].value}`)}`;
    } else {
      res += `:${chalk.white.underline(tag)}`;
    }
  }
  return res;
};

const exec = (isTest, tag, bin, cb) => {
  if (isTest) {
    debug(`${tagger(tag)} run ${bin}`);
    if (cb) {
      cb(null);
    }
  } else {
    let spin = ora(`${bin} running ...\n`).start();
    let start = Date.now();
    pexec(bin, {
      maxBuffer: 5000 * 1024
    })
      .then(res => {
        spin.succeed(
          `${bin} succeed ${chalk.white(
            `[${Math.floor((Date.now() - start) / 1000)}s]`
          )}\n`
        );
        debug(`${tagger(tag)} run ${bin} output: ${res.stdout}`);
        if (cb) {
          cb(null, res.stdout);
        }
      })
      .catch(err => {
        spin.fail(`${bin} fail\n`);
        debug(`${tagger(tag)} run ${bin} err: ${err.toString()}`);
        if (cb) {
          cb(err);
        }
      });
  }
};

const sexec = (isTest, tag, bin, cb) => {
  if (isTest) {
    debug(`${tagger(tag)} run ${bin}`);
    if (cb) {
      cb(null);
    }
  } else {
    pexec(bin, {
      maxBuffer: 500 * 1024
    })
      .then(res => {
        debug(`${tagger(tag)} run ${bin} output: ${res.stdout}`);
        if (cb) {
          cb(null, res.stdout);
        }
      })
      .catch(err => {
        debug(`${tagger(tag)} run ${bin} err: ${err.toString()}`);
      });
  }
};

const versionInfo = () => {
  console.log(chalk.white.bold(`----------------------`));
  console.log(chalk.white.bold(`MARO Version: ${version}`));
  console.log(chalk.white.bold(`----------------------`));
};

const help = gulp => {
  versionInfo();
  art
    .font('MARO', 'Doom')
    .toPromise()
    .then(rendered => {
      let ani = chalkAni.rainbow(rendered, 3);
      setTimeout(() => {
        ani.stop();
        return usage(gulp);
      }, 600);
    });
};

const notify = (spin, start, watchPath, cb) => {
  watch(watchPath)
    .on('change', file => {
      cb(file);
    })
    .on('add', file => {
      cb(file);
    })
    .on('unlink', file => {
      cb(file);
    });

  setInterval(() => {
    if (spin.isSpinning) {
      spin.text = `watching ... ${Math.floor(
        (Date.now() - start) / 1000 / 3600
      )}h ${Math.floor((Date.now() - start) / 1000 / 60) % 60}m ${Math.floor(
        (Date.now() - start) / 1000
      ) % 60}s`;
    }
  }, 1000);
};

const writeDocs = isTest => {
  let start = Date.now();
  debug(
    chalk.white(
      `=========================================================================================`
    )
  );
  debug(
    chalk.white(
      `Begin watching your change/add/del in docs, will auto-build/deploy for these events ...`
    )
  );
  debug(
    chalk.white(
      `=========================================================================================`
    )
  );
  let spin = ora(`watching ...\n`).start();
  const rebuild = file => {
    sexec(isTest, `local`, `cd ../docs; make html`, (err, res) => {
      console.log(err, res);
      let now = new Date();
      now.setHours(now.getHours() + 8);
      if (!err) {
        spin.succeed(
          `${dateFormat(now, 'dddd, mmmm dS, yyyy, h:MM:ss TT')} ${chalk.yellow(
            `[change] ${file} rebuild finished.`
          )}`
        );
      } else {
        spin.fail(
          `${dateFormat(now, 'dddd, mmmm dS, yyyy, h:MM:ss TT')} ${chalk.yellow(
            `[change] ${file} rebuild failed: `
          )}${chalk.white.bold.italic(`${err}`)}`
        );
      }
      spin.start(
        `watching ... ${Math.floor(
          (Date.now() - start) / 1000 / 3600
        )}h ${Math.floor((Date.now() - start) / 1000 / 60) % 60}m ${Math.floor(
          (Date.now() - start) / 1000
        ) % 60}s`
      );
    });
  };
  notify(spin, start, '../docs/**/*.py', rebuild);
  notify(spin, start, '../docs/source/**/*.md', rebuild);
  notify(spin, start, '../docs/source/**/*.rst', rebuild);
  notify(spin, start, '../docs/source/**/*.png', rebuild);
};

const buildDocs = isTest => {
  sexec(isTest, `local`, `cd ../docs; make html`);
};

const hostDocs = (isTest, isSync) => {
  inquirer
    .prompt([{
      type: 'input',
      name: 'port',
      message: `What's the port for readthedocs server?`,
      default: '40010'
    }])
    .then(input => {
      publicIp.v4().then(ip => {
        console.log(chalk.yellow.bold(`Host on: ${ip}:${input.port}`));
        let hostDocsBin = '';
        if (isSync) {
          hostDocsBin = `cd ../docs/_build/html; browser-sync start --server --port ${input.port} --watch --files "**/*"`;
        } else {
          hostDocsBin = `cd ../docs/_build/html; python -m http.server ${input.port}`;
        }
        // console.log(`${chalk.white('copy and run:')} ${chalk.blue(hostDocsBin)}`);
        exec(
          isTest,
          `local|${os.userInfo().username}@${os.hostname()}`,
          hostDocsBin
        );
      });
    })
    .catch(debug);
};

const md2Rst = isTest => {
  inquirer
    .prompt([{
      type: 'input',
      name: 'folder',
      message: `What's the folder for converting?`,
      default: '.'
    }])
    .then(input => {
      let convertBin = `cd ${
        input.folder
        }; bash ${process.cwd()}/bin/md2rst.sh`;
      exec(isTest, `local`, convertBin);
    })
    .catch(debug);
};

const buildSim = isTest => {
  inquirer
    .prompt([{
      type: 'input',
      name: 'projectRoot',
      message: `What's the project root path?`,
      default: `${path.resolve(__dirname, '..')}`
    }])
    .then(input => {
      let buildSim = null;

      if (process.platform == 'win32') {
        buildSim = 'TODO @chaos';
      } else {
        buildSim = `PROJECT_ROOT=${input.projectRoot} bash ${path.join(
          __dirname,
          'scripts',
          'build_sim.sh'
        )}`;
      }

      exec(isTest, `local`, buildSim);
    })
    .catch(debug);
};

module.exports = {
  exec: exec,
  sexec: sexec,
  help: help,
  versionInfo: versionInfo,
  writeDocs: writeDocs,
  hostDocs: hostDocs,
  buildDocs: buildDocs,
  md2Rst: md2Rst,
  buildSim: buildSim
};