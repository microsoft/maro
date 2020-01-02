# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import logging
import random
import time
from functools import wraps

from maro.distributed import Proxy

_ALIAS_MAP = {'environment_runner': 'ENV', 'learner': 'LRN'}


def get_proxy(component_type, cfg, logger=None):
    """
    Generate proxy by given component_type and config

    Args:
        component_type: str
        cfg: dottable_dict
        logger: logger object
    Return:
        Proxy: Class
    """
    comp = cfg.distributed[component_type]

    def get_audience():
        if 'audience' not in comp:
            return

        audience = []
        for peer in comp.audience:
            audi = cfg.distributed[peer]
            peer_cnt = int(audi.num)
            if peer_cnt > 1:
                audience.extend(['_'.join([peer, str(i)]) for i in range(peer_cnt)])
            else:
                audience.append(peer)

        return audience

    return Proxy(receive_enabled=comp.receive_enabled, audience_list=get_audience(),
                 redis_host=cfg.redis.host, redis_port=cfg.redis.port, logger=logger)


def log(logger):
    def handle_with_log(handler_fn):
        @wraps(handler_fn)
        def handler_decorator(*args):
            msg = args[2]
            logger.info(f'received a {msg.type.name} message from {msg.src}')
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
