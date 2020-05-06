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
                # use functools.partial to freeze handling function's local_instance and proxy
                # arguments to self.local_instance and self.proxy
                for idx, info in enumerate(handler_dict):
                    self.handler_dict = {idx: partial(info['handler_fn'], self.local_instance, self.proxy)}
                    self.msg_request = {idx: info['request']}

            def __getattr__(self, name):
                if name in self.__dict__:
                    return self.__dict__[name]

                return getattr(self.local_instance, name)

            def launch(self):
                """
                Universal entry point for running a cls instance in distributed mode
                """
                self.proxy.join()
                self.proxy.add_msg_request(self.msg_request)
                for msg in self.proxy.receive():
                    request_idx, n_msg = self.proxy.msg_handler(msg)
                    if n_msg is not None:
                        self.handler_dict[request_idx](n_msg)

        return Wrapper

    return dist_decorator
