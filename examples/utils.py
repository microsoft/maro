# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import logging
import random
import time
from functools import wraps


def get_peers(component_type, config_dict):
    """
    Generate a complete list of peer names for a component from configuration
    """
    if 'peers' not in config_dict[component_type]:
        return

    peers = []
    for peer_type in config_dict[component_type].peers:
        count = int(config_dict[peer_type].num)
        if count > 1:
            peers.extend(['_'.join([peer_type, str(i)]) for i in range(count)])
        else:
            peers.append(peer_type)

    return peers


def log(logger):
    def handle_with_log(handler_fn):
        @wraps(handler_fn)
        def handler_decorator(*args):
            msg = args[2]
            logger.info(f'received a {msg.type.name} message from {msg.source}')
            handler_fn(*args)

        return handler_decorator

    return handle_with_log


def generate_random_rgb():
    return f'rgb({random.randint(0,255)}, {random.randint(0,255)}, {random.randint(0,255)})'


HTML_FORMAT = '<p><pre style="color:%(color)s;">[%(time)s]-[%(name)s] %(msg)s</pre></p>'


class HTMLFormatter(logging.Formatter):
    def __init__(self, fmt, display_name=None, **kwargs):
        super().__init__(fmt=fmt)
        self._start_time = time.time()
        self._display_name = display_name
        for k, v in kwargs.items():
            setattr(self, '_'+k, v)

    def set_display_name(self, name):
        self._display_name = name

    def format(self, record):
        t = time.time() - self._start_time
        return self._fmt % {'time': '{:10.1f}'.format(t), 'msg': record.msg,
                            'color': self._color, 'name': '{:^23}'.format(self._display_name.upper())}


class HTMLLogger:
    def __init__(self, file_name, write_mode='a', **kwargs):
        self._logger = logging.getLogger()
        self._logger.setLevel(logging.DEBUG)
        self._html_formatter = HTMLFormatter(fmt=HTML_FORMAT, **kwargs)
        fh = logging.FileHandler(file_name, mode=write_mode)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(self._html_formatter)
        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)
        sh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        sh.setFormatter(sh_formatter)
        self._logger.addHandler(fh)
        self._logger.addHandler(sh)

    def set_display_name(self, name):
        self._html_formatter.set_display_name(name)

    def debug(self, msg):
        self._logger.debug(msg)

    def info(self, msg):
        self._logger.info(msg)

    def warn(self, msg):
        self._logger.warn(msg)

    def error(self, msg):
        self._logger.error(msg)

    def critical(self, msg):
        self._logger.critical(msg)
