// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.


// native lib
const os = require('os');

// 3rd party lib

// private lib
const util = require(`${__dirname}/util`);

const buildImage = (isTest) => {
    const buildScript = `cd ../maro/utils/dashboard/dashboard_resource; bash ./build.sh; cd -`;
    util.exec(isTest, `local|${os.userInfo().username}@${os.hostname()}`, buildScript);
};

const startService = (isTest) => {
    const startScript = `cd ../maro/utils/dashboard/dashboard_resource; bash ./start.sh; cd -`;
    util.exec(isTest, `local|${os.userInfo().username}@${os.hostname()}`, startScript);
};

const stopService = (isTest) => {
    const stopScript = `cd ../maro/utils/dashboard/dashboard_resource; bash ./stop.sh; cd -`;
    util.exec(isTest, `local|${os.userInfo().username}@${os.hostname()}`, stopScript);
};


module.exports = {
    buildImage: buildImage,
    startService: startService,
    stopService: stopService,
};