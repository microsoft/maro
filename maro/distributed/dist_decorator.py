# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# private lib
from maro.distributed.proxy import Proxy
from typing import Callable
from functools import partial


def dist(proxy: Proxy, handler_dict: {object: Callable}):
    """
    A decorator used to inject a communication module and message handlers
    to a local class so that it can be run in distributed mode.
    """
    def dist_decorator(cls):
        class Wrapper:
            """
            A wrapper class for cls, the class to be decorated. It contains a reference
            to the proxy and a message handler lookup table and defines a launch method
            as the universal entry point for running a cls instance in distributed mode
            """
            def __init__(self, *args, **kwargs):
                self.local_instance = cls(*args, **kwargs)
                self.proxy = proxy
                self._msg_request_handle_list = []
                # use functools.partial to freeze handling function's local_instance and proxy
                # arguments to self.local_instance and self.proxy
                for idx, info in enumerate(handler_dict):
                    self._msg_request_handle_list.append({'request': info['request'],
                                                          'remain': info['request'], 
                                                          'msg_list': [],
                                                          'handler_fn': partial(info['handler_fn'], self.local_instance, self.proxy)})

            def __getattr__(self, name):
                if name in self.__dict__:
                    return self.__dict__[name]

                return getattr(self.local_instance, name)

            def launch(self):
                """
                Universal entry point for running a cls instance in distributed mode
                """
                self.proxy.join()
                for msg in self.proxy.receive():
                    for req_dict in self._msg_request_handle_list:
                        for key, value in req_dict['remain'].items():
                            if key == (msg.source, msg.type) or key == (True, msg.type) or key == (msg.source, True):
                                if value > 1:
                                    req_dict['remain'].update({key: value - 1})
                                else:
                                    del req_dict['remain'][key]
                                req_dict['msg_list'].append(msg)

                                if not req_dict['remain']:
                                    request_msg_list = req_dict['msg_lst'][:]
                                    req_dict['msg_lst'] = []
                                    req_dict['remain'] = req_dict['request']
                                    req_dict['handler_fn'](request_msg_list)
                                
                    raise Exception(f"Unexpected Msg, which msg_type is {msg.type} and source is {msg.source}")

        return Wrapper

    return dist_decorator
