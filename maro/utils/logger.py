# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from datetime import datetime
from enum import Enum
import getpass
import logging
import os
import socket
import sys

cwd = os.getcwd()

# for api generation, we should hide our build path for security issue
if "APIDOC_GEN" in os.environ:
    cwd = ""


class LogFormat(Enum):
    """
    Format of log
    """
    full = 1
    simple = 2
    none = 3


format_store = {
    LogFormat.full: logging.Formatter(
        fmt='%(asctime)s | %(host)s | %(user)s | %(process)d | %(tag)s | %(levelname)s | %(message)s'),
    LogFormat.simple: logging.Formatter(
        fmt='%(asctime)s | %(tag)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S'),
    LogFormat.none: None
}

PROGRESS = 60  # progress of training, we give it a highest level
logging.addLevelName(PROGRESS, "PROGRESS")

level_map = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARN,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
    "PROGRESS": PROGRESS
}


def msgformat(logfunc):
    """decorator to construct the log msg"""

    def _msgformatter(self, msg, *args):
        if args:
            logfunc(self, "%s %s", isinstance(msg, str)
                    and msg or repr(msg), repr(args))
        else:
            logfunc(self, "%s", isinstance(msg, str)
                    and msg or repr(msg))

    return _msgformatter


class Logger:
    """A simple wrapper for logging, the console logging level can be set by environment variable, which
        also can be redirected.
        e.g. export LOG_LEVEL=DEBUG.
        Supported log level:
                            DEBUG
                            INFO
                            WARN
                            ERROR
                            CRITICAL
                            PROGRESS
        the file logging level is set to DEBUG, which cannot be impacted by the LOG_LEVEL.

    Args:
        tag (str): Log tag for stream and file output.
        format_ (LogFormat): Predefined formatter, the default value is LogFormat.full.
                        i.e. LogFormat.full: full time | host | user | pid | tag | level | msg
                             LogFormat.simple: simple time | tag | level | msg
        dump_folder (str): Log dumped folder, the default value is the current folder. The dumped log level is logging.DEBUG.
                        The full path of the dumped log file is `dump_folder/tag.log`.
        dump_mode (str): Write log file mode, the default value is 'w'. For appending, please use 'a'.
        extension_name (str): Final dumped file extension name, default value is 'log'.
        auto_timestamp (bool): If true the dumped log file name will add a timestamp. (e.g. tag.1574953673.137387.log)
    """

    def __init__(self, tag: str, format_: LogFormat = LogFormat.full, dump_folder: str = cwd, dump_mode: str = 'w',
                 extension_name: str = 'log', auto_timestamp: bool = True):
        self._format = format_store[format_]
        self._level = os.environ.get('LOG_LEVEL') or "DEBUG"
        self._logger = logging.getLogger(tag)
        self._logger.setLevel(logging.DEBUG)
        self._extension_name = extension_name

        if not os.path.exists(dump_folder):
            os.makedirs(dump_folder)

        if auto_timestamp:
            filename = f'{tag}.{datetime.now().timestamp()}'
        else:
            filename = f'{tag}'

        filename += f'.{self._extension_name}'

        fh = logging.FileHandler(
            filename=f'{os.path.join(dump_folder, filename)}', mode=dump_mode)

        fh.setLevel(logging.DEBUG)

        if self._format is not None:
            fh.setFormatter(self._format)

        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(self._level)

        if self._format is not None:
            sh.setFormatter(self._format)

        self._logger.addHandler(fh)
        self._logger.addHandler(sh)

        self._extra = {'host': socket.gethostname(), 'user': getpass.getuser(), 'tag': tag}

    @msgformat
    def debug(self, msg, *args):
        self._logger.debug(msg, *args, extra=self._extra)

    @msgformat
    def info(self, msg, *args):
        self._logger.info(msg, *args, extra=self._extra)

    @msgformat
    def warn(self, msg, *args):
        self._logger.warning(msg, *args, extra=self._extra)

    @msgformat
    def error(self, msg, *args):
        self._logger.error(msg, *args, extra=self._extra)

    @msgformat
    def critical(self, msg, *args):
        self._logger.critical(msg, *args, extra=self._extra)


if __name__ == '__main__':
    full_logger = Logger(tag='full_format', dump_mode='w')
    for i in range(100):
        full_logger.debug('debug msg')
        full_logger.info('info msg')
        full_logger.warn('warn msg')
        full_logger.error('error msg')
        full_logger.critical('critical msg')

    simple_logger = Logger(tag='simple_format',
                           format=LogFormat.simple, dump_mode='w')
    for i in range(100):
        simple_logger.debug('debug msg')
        simple_logger.info('info msg')
        simple_logger.warn('warn msg')
        simple_logger.error('error msg')
        simple_logger.critical('critical msg')
