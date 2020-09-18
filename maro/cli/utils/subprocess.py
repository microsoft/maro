# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import subprocess

from maro.utils.exception.cli_exception import CommandError
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


class SubProcess:
    @staticmethod
    def run(command: str) -> str:
        logger.debug(command)
        completed_process = subprocess.run(command,
                                           shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8')
        if completed_process.returncode != 0:
            raise CommandError(command, completed_process.stderr)
        return completed_process.stdout

    @staticmethod
    def interactive_run(command: str) -> None:
        logger.debug(command)
        process = subprocess.Popen(command,
                                   executable='/bin/bash',
                                   shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8')
        while True:
            next_line = process.stdout.readline()
            if next_line == '' and process.poll() is not None:
                break
            logger.debug(next_line.strip('\n'))
        stdout, stderr = process.communicate()
        if stderr:
            logger.debug_yellow(stderr.strip('\n'))
