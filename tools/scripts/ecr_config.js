// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.


// native lib
const os = require('os');
const path = require('path');

// 3rd party lib
const chalk = require('chalk');
const debug = require('debug')('docker');
const inquirer = require('inquirer');

// private lib
const util = require(`${__dirname}/util`);

const generateConfig = (isTest) => {
    let generateScript = 'python scripts/ecr_config_utils/config_auto_generator.py';
    util.exec(isTest, `local|${os.userInfo().username}@${os.hostname()}`, generateScript);
}

const visualizeOrder = (isTest) => {
    let visualizeScript = 'python scripts/ecr_config_utils/order_auto_visualizer.py';
    util.exec(isTest, `local|${os.userInfo().username}@${os.hostname()}`, visualizeScript);
}

const visualizeTopology = (isTest) => {
    let visualizeScript = 'python scripts/ecr_config_utils/topo_auto_visualizer.py';
    util.exec(isTest, `local|${os.userInfo().username}@${os.hostname()}`, visualizeScript);
}

module.exports = {
    generateConfig: generateConfig,
    visualizeOrder: visualizeOrder,
    visualizeTopology: visualizeTopology
};
