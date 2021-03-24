# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# native lib
import getpass
import logging
import os
import socket
import sys
from datetime import datetime
from enum import Enum

# private lib
from maro.cli.utils.params import GlobalParams as CliGlobalParams

cwd = os.getcwd()

# For API generation, we should hide our build path for security issue.
if "APIDOC_GEN" in os.environ:
    cwd = ""


class LogFormat(Enum):
    """The Enum class of the log format.

    Example:
        - ``LogFormat.full``: full time | host | user | pid | tag | level | msg
        - ``LogFormat.simple``: simple time | tag | level | msg
        - ``LogFormat.cli_debug`` : simple time | level | msg
        - ``LogFormat.cli_info`` (file): simple time | level | msg
        - ``LogFormat.cli_info`` (stdout):  msg
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
        fmt="%(asctime)s | %(levelname)-7s | %(threadName)-10s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"),
    LogFormat.cli_info: logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(threadName)-10s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"),
    LogFormat.none: None
}

FORMAT_NAME_TO_STDOUT_FORMAT = {
    # We need to output clean messages in the INFO mode.
    LogFormat.cli_info: logging.Formatter(fmt='%(message)s'),
}

# Progress of training, we give it a highest level.
PROGRESS = 60
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
    """The decorator used to construct the log msg."""

    def _msgformatter(self, msg, *args):
        if args:
            logfunc(self, "%s %s", isinstance(msg, str) and msg or repr(msg), repr(args))
        else:
            logfunc(self, "%s", isinstance(msg, str) and msg or repr(msg))

    return _msgformatter


class Logger(object):
    """A simple wrapper for logging.

    The Logger hosts a file handler and a stdout handler. The file handler is set
    to ``DEBUG`` level and will dump all the logging info to the given ``dump_folder``.
    The logging level of the stdout handler is decided by the ``stdout_level``,
    and can be redirected by setting the environment variable ``LOG_LEVEL``.
    Supported ``LOG_LEVEL`` includes: ``DEBUG``, ``INFO``, ``WARN``, ``ERROR``,
    ``CRITICAL``, ``PROCESS``.

    Example:
        ``$ export LOG_LEVEL=INFO``

    Args:
        tag (str): Log tag for stream and file output.
        format_ (LogFormat): Predefined formatter. Defaults to ``LogFormat.full``.
        dump_folder (str): Log dumped folder. Defaults to the current folder.
            The dumped log level is ``logging.DEBUG``. The full path of the
            dumped log file is `dump_folder/tag.log`.
        dump_mode (str): Write log file mode. Defaults to ``w``. Use ``a`` to
            append log.
        extension_name (str): Final dumped file extension name. Defaults to `log`.
        auto_timestamp (bool): Add a timestamp to the dumped log file name or not.
            E.g: `tag.1574953673.137387.log`.
        stdout_level (str): the logging level of the stdout handler. Defaults to
            ``DEBUG``.
    """

    def __init__(
        self, tag: str, format_: LogFormat = LogFormat.simple, dump_folder: str = cwd, dump_mode: str = 'w',
        extension_name: str = 'log', auto_timestamp: bool = False, stdout_level="INFO"
    ):
        self._file_format = FORMAT_NAME_TO_FILE_FORMAT[format_]
        self._stdout_format = FORMAT_NAME_TO_STDOUT_FORMAT[format_] \
            if format_ in FORMAT_NAME_TO_STDOUT_FORMAT else \
            FORMAT_NAME_TO_FILE_FORMAT[format_]
        self._stdout_level = os.environ.get('LOG_LEVEL') or stdout_level
        self._logger = logging.getLogger(tag)
        self._logger.setLevel(logging.DEBUG)
        self._extension_name = extension_name

        if not os.path.exists(dump_folder):
            try:
                os.makedirs(dump_folder)
            except FileExistsError:
                logging.warning("Receive File Exist Error about creating dump folder for internal log. "
                                "It may be caused by multi-thread and it won't have any impact on logger dumps.")
            except Exception as e:
                raise e

        if auto_timestamp:
            filename = f'{tag}.{datetime.now().timestamp()}'
        else:
            filename = f'{tag}'

        filename += f'.{self._extension_name}'

        # File handler
        fh = logging.FileHandler(filename=f'{os.path.join(dump_folder, filename)}', mode=dump_mode, encoding="utf-8")
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
        """Add a log with ``DEBUG`` level."""
        self._logger.debug(msg, *args, extra=self._extra)

    @msgformat
    def info(self, msg, *args):
        """Add a log with ``INFO`` level."""
        self._logger.info(msg, *args, extra=self._extra)

    @msgformat
    def warn(self, msg, *args):
        """Add a log with ``WARN`` level."""
        self._logger.warning(msg, *args, extra=self._extra)

    @msgformat
    def error(self, msg, *args):
        """Add a log with ``ERROR`` level."""
        self._logger.error(msg, *args, extra=self._extra)

    @msgformat
    def critical(self, msg, *args):
        """Add a log with ``CRITICAL`` level."""
        self._logger.critical(msg, *args, extra=self._extra)


class DummyLogger:
    """A dummy Logger, which is used when disabling logs."""

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


class InternalLogger(Logger):
    """An internal logger uses for recording the internal system's log."""

    def __init__(
        self, component_name: str, tag: str = "maro_internal", format_: LogFormat = LogFormat.internal,
        dump_folder: str = None, dump_mode: str = 'a', extension_name: str = 'log',
        auto_timestamp: bool = False
    ):
        current_time = f"{datetime.now().strftime('%Y%m%d%H%M')}"
        self._dump_folder = dump_folder if dump_folder else \
            os.path.join(os.path.expanduser("~"), ".maro/log", current_time, str(os.getpid()))
        super().__init__(tag, format_, self._dump_folder, dump_mode, extension_name, auto_timestamp)

        self._extra = {'component': component_name}


class CliLogger:
    """An internal logger for CLI logging.

    It maintains a singleton logger in a CLI command lifecycle.
    The logger is inited at the very beginning, and use different logging formats based on the ``--debug`` argument.
    """

    class _CliLogger(Logger):
        def __init__(self):
            """Init singleton logger based on the ``--debug`` argument."""
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

    def passive_init(self) -> None:
        """Init a new ``CliLogger`` if current logger is not matched with the parameters."""
        if not CliLogger._logger or CliLogger._logger.log_level != CliGlobalParams.LOG_LEVEL:
            CliLogger._logger = self._CliLogger()

    def debug(self, message: str) -> None:
        """``logger.debug()`` with passive init.

        Args:
            message (str): logged message.
        """
        self.passive_init()
        self._logger.debug(message)

    def debug_yellow(self, message: str) -> None:
        """``logger.debug()`` with color yellow and passive init.

        Args:
            message (str): logged message.
        """
        self.passive_init()
        self._logger.debug('\033[33m' + message + '\033[0m')

    def debug_green(self, message: str) -> None:
        """``logger.debug()`` with color green and passive init.

        Args:
            message (str): logged message.
        """
        self.passive_init()
        self._logger.debug('\033[32m' + message + '\033[0m')

    def info(self, message: str) -> None:
        """``logger.info()`` with passive init.

        Args:
            message (str): logged message.
        """
        self.passive_init()
        self._logger.info(message)

    def warning(self, message: str) -> None:
        """``logger.warning()`` with passive init.

        Args:
            message (str): logged message.
        """
        self.passive_init()
        self._logger.warn(message)

    def error(self, message: str) -> None:
        """``logger.error()`` with passive init.

        Args:
            message (str): logged message.
        """
        self.passive_init()
        self._logger.error(message)

    def info_green(self, message: str) -> None:
        """``logger.info()`` with color green and passive init.

        Args:
            message (str): logged message.
        """
        self.passive_init()
        self._logger.info('\033[32m' + message + '\033[0m')

    def warning_yellow(self, message: str) -> None:
        """``logger.warning()`` with color yellow and passive init.

        Args:
            message (str): logged message.
        """
        self.passive_init()
        self._logger.warn('\033[33m' + message + '\033[0m')

    def error_red(self, message: str) -> None:
        """``logger.error()`` with color red and passive init.

        Args:
            message (str): logged message.
        """
        self.passive_init()
        self._logger.error('\033[31m' + message + '\033[0m')
