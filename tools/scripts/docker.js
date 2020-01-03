// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.


// native lib
const fs = require('fs');
const os = require('os');
const path = require('path');

// 3rd party lib
const chalk = require('chalk');
const debug = require('debug')('docker');
const inquirer = require('inquirer');

// private lib
const util = require(`${__dirname}/util`);

const buildImage = (isTest) => {
    const dockerFilesDir = `${path.resolve(__dirname, '..', '..', 'docker_files')}`;
    const dockerFileList = [];

    fs.readdirSync(dockerFilesDir).forEach((file) => {
        if (file.split('.').slice(-1)[0] == 'df') {
            dockerFileList.unshift(`${dockerFilesDir}/${file}`);
        }
    });

    if (dockerFileList.length == 0) {
        console.log(chalk.red(`Create your docker files in ${dockerFilesDir} firstly!`));
        process.exit(1);
    }

    inquirer.prompt([{
                type: 'list',
                name: 'dockerFile',
                message: 'Choose your docker file.',
                choices: dockerFileList,
                default: 0
            },
            {
                type: 'input',
                name: 'imageName',
                message: `What's the docker image name for building?`,
                default: 'maro/dev/cpu'
            }
        ])
        .then((input) => {
            const dockerFile = input.dockerFile;
            const buildFolder = '..';
            const dockerImageName = input.imageName;
            const buildScript = `DOCKER_FILE=${dockerFile} DOCKER_FILE_DIR=${buildFolder} DOCKER_IMAGE_NAME=${dockerImageName} bash ./scripts/build_image.sh`;
            util.exec(isTest, `local|${os.userInfo().username}@${os.hostname()}`, buildScript);
        })
        .catch(debug);
};

const launchContainer = (isTest) => {
    inquirer.prompt([{
                type: 'input',
                name: 'imageName',
                message: `What's the docker image name for launching?`,
                default: 'maro/dev/cpu'
            },
            {
                type: 'input',
                name: 'containerName',
                message: `What's the docker container name for launching?`,
                default: 'ecr_arthur'
            }
        ])
        .then((input) => {
            let launchScript = `DOCKER_IMAGE_NAME=${input.imageName} DOCKER_CONTAINER_NAME=${input.containerName} MOUNT_PATH=${path.resolve(__dirname, '..', '..')} bash ./scripts/run_container.sh`;
            util.exec(isTest, `local|${os.userInfo().username}@${os.hostname()}`, launchScript);
        })
        .catch(debug);
};

const loginContainer = (isTest) => {
    inquirer.prompt([{
            type: 'input',
            name: 'containerName',
            message: `What's the docker container name for login?`,
            default: 'ecr_arthur'
        }])
        .then((input) => {
            let loginBin = `DOCKER_CONTAINER_NAME=${input.containerName} bash ./scripts/login_container.sh`;
            console.log(chalk.blue(`copy and run: ${chalk.green(`${loginBin}`)}`));
        })
        .catch(debug);
};

const cleanContainer = (isTest, isForce) => {
    let cleanScript = `docker ps -a | egrep "Exited|Created" | cut -d" " -f1 | xargs sudo docker stop -t 0 | xargs sudo docker rm`;

    if (isForce) {
        cleanScript = `docker ps -a | cut -d" " -f1 | xargs sudo docker stop -t 0 | xargs sudo docker rm`;
    }

    util.exec(isTest, `local|${os.userInfo().username}@${os.hostname()}`, cleanScript);
}

const cleanImage = (isTest) => {
    let cleanScript = 'docker images -q | xargs sudo docker image rm -f';
    util.exec(isTest, `local|${os.userInfo().username}@${os.hostname()}`, cleanScript);
}

const listImage = (isTest) => {
    let listImageScript = 'docker image ls';
    util.exec(isTest, `local|${os.userInfo().username}@${os.hostname()}`, listImageScript);
}

const listContainer = (isTest) => {
    let listContainerScript = 'docker ps -a';
    util.exec(isTest, `local|${os.userInfo().username}@${os.hostname()}`, listContainerScript);
}

module.exports = {
    buildImage: buildImage,
    launchContainer: launchContainer,
    loginContainer: loginContainer,
    cleanContainer: cleanContainer,
    cleanImage: cleanImage,
    listImage: listImage,
    listContainer: listContainer
};