# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import getpass
import logging
import os
import socket
import sys
from datetime import datetime
from enum import Enum

from maro.cli.utils.params import GlobalParams as CliGlobalParams

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
    internal = 3
    cli_debug = 4
    cli_info = 5
    none = 6


FORMAT_NAME_TO_FILE_FORMAT = {
    LogFormat.full: logging.Formatter(
        fmt='%(asctime)s | %(host)s | %(user)s | %(process)d | %(tag)s | %(levelname)s | %(message)s'),
    LogFormat.simple: logging.Formatter(
        fmt='%(asctime)s | %(tag)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S'),
    LogFormat.internal: logging.Formatter(
        fmt='%(asctime)s | %(component)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S'),
    LogFormat.cli_debug: logging.Formatter(
        fmt='%(asctime)s | %(levelname)-7s | %(message)s', datefmt='%H:%M:%S'),
    LogFormat.cli_info: logging.Formatter(
        fmt='%(asctime)s | %(levelname)-7s | %(message)s', datefmt='%H:%M:%S'),
    LogFormat.none: None
}

FORMAT_NAME_TO_STDOUT_FORMAT = {
    LogFormat.cli_info: logging.Formatter(fmt='%(message)s'),
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
    """
    A simple wrapper for logging, the console logging level can be set by environment variable, which
    also can be redirected.

    e.g. export LOG_LEVEL=DEBUG.

    Supported log levels
        - DEBUG
        - INFO
        - WARN
        - ERROR
        - CRITICAL
        - PROGRESS

    Note: The file logging level is set to DEBUG, which cannot be impacted by the LOG_LEVEL.

    Args:
        tag (str): Log tag for stream and file output.
        format_ (LogFormat): Predefined formatter, the default value is LogFormat.full.
                        i.e. LogFormat.full: full time | host | user | pid | tag | level | msg
                             LogFormat.simple: simple time | tag | level | msg
        dump_folder (str): Log dumped folder, the default value is the current folder. The dumped log level is
                        logging.DEBUG. The full path of the dumped log file is `dump_folder/tag.log`.
        dump_mode (str): Write log file mode, the default value is 'w'. For appending, please use 'a'.
        extension_name (str): Final dumped file extension name, default value is 'log'.
        auto_timestamp (bool): If true the dumped log file name will add a timestamp. (e.g. tag.1574953673.137387.log)
    """

    def __init__(self, tag: str, format_: LogFormat = LogFormat.full, dump_folder: str = cwd, dump_mode: str = 'w',
                 extension_name: str = 'log', auto_timestamp: bool = True, stdout_level="INFO"):
        self._file_format = FORMAT_NAME_TO_FILE_FORMAT[format_]
        self._stdout_format = FORMAT_NAME_TO_STDOUT_FORMAT[format_] \
            if format_ in FORMAT_NAME_TO_STDOUT_FORMAT else \
            FORMAT_NAME_TO_FILE_FORMAT[format_]
        self._stdout_level = os.environ.get('LOG_LEVEL') or stdout_level
        self._logger = logging.getLogger(tag)
        self._logger.setLevel(logging.INFO)
        self._extension_name = extension_name

        if not os.path.exists(dump_folder):
            try:
                os.makedirs(dump_folder)
            except FileExistsError as e:
                logging.warning(f"Receive File Exist Error about creating dump folder for internal log. "
                                f"It may be caused by multi-thread and it won't have any impact on logger dumps.")
            except Exception as e:
                raise e

        if auto_timestamp:
            filename = f'{tag}.{datetime.now().timestamp()}'
        else:
            filename = f'{tag}'

        filename += f'.{self._extension_name}'

        # File handler
        fh = logging.FileHandler(
            filename=f'{os.path.join(dump_folder, filename)}', mode=dump_mode)
        fh.setLevel(logging.DEBUG)
        if self._file_format is not None:
            fh.setFormatter(self._file_format)

        # Stdout handler
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(self._stdout_level)
        if self._stdout_format is not None:
            sh.setFormatter(self._stdout_format)

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


class InternalLogger(Logger):
    """ An internal logger uses for record internal system's log """

    def __init__(self, component_name: str, tag: str = "maro_internal", format_: LogFormat = LogFormat.internal,
                 dump_folder: str = None, dump_mode: str = 'a', extension_name: str = 'log',
                 auto_timestamp: bool = False):
        current_time = f"{datetime.now().strftime('%Y%m%d%H%M')}"
        self._dump_folder = dump_folder if dump_folder else \
            os.path.join(os.path.expanduser("~"), ".maro/log", current_time, str(os.getpid()))
        super().__init__(tag, format_, self._dump_folder, dump_mode, extension_name, auto_timestamp)

        self._extra = {'component': component_name}


class DummyLogger:
    def __init__(self):
        pass

    def debug(self, msg, *args):
        pass

    def info(self, msg, *args):
        pass

    def warn(self, msg, *args):
        pass

    def error(self, msg, *args):
        pass

    def critical(self, msg, *args):
        pass


class CliLogger:
    class _CliLogger(Logger):
        def __init__(self):
            # Init params
            self.log_level = CliGlobalParams.LOG_LEVEL

            current_time = f"{datetime.now().strftime('%Y%m%d')}"
            self._dump_folder = os.path.join(os.path.expanduser("~/.maro/log/cli"), current_time)
            if self.log_level == logging.DEBUG:
                super().__init__(
                    tag='cli',
                    format_=LogFormat.cli_debug, dump_folder=self._dump_folder,
                    dump_mode='a', extension_name='log', auto_timestamp=False, stdout_level=self.log_level
                )
            elif self.log_level >= logging.INFO:
                super().__init__(
                    tag='cli',
                    format_=LogFormat.cli_info, dump_folder=self._dump_folder,
                    dump_mode='a', extension_name='log', auto_timestamp=False, stdout_level=self.log_level
                )

    _logger = None

    def __init__(self, name):
        self.name = name

    def passive_init(self):
        if not CliLogger._logger or CliLogger._logger.log_level != CliGlobalParams.LOG_LEVEL:
            CliLogger._logger = CliLogger._CliLogger()

    def debug(self, message: str):
        self.passive_init()
        self._logger.debug(message)

    def debug_yellow(self, message: str):
        self.passive_init()
        self._logger.debug('\033[33m' + message + '\033[0m')

    def info(self, message: str):
        self.passive_init()
        self._logger.info(message)

    def warning(self, message: str):
        self.passive_init()
        self._logger.warn(message)

    def error(self, message: str):
        self.passive_init()
        self._logger.error(message)

    def info_green(self, message: str):
        self.passive_init()
        self._logger.info('\033[32m' + message + '\033[0m')

    def warning_yellow(self, message: str):
        self.passive_init()
        self._logger.warn('\033[33m' + message + '\033[0m')

    def error_red(self, message: str):
        self.passive_init()
        self._logger.error('\033[31m' + message + '\033[0m')
