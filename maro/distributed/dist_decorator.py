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
                # use functools.partial to freeze handling function's local_instance and proxy
                # arguments to self.local_instance and self.proxy
                self.handler_dict = {msg_type: partial(handler_fn, self.local_instance, proxy)
                                     for msg_type, handler_fn in handler_dict.items()}

            def __getattr__(self, name):
                if name in self.__dict__:
                    return self.__dict__[name]

                return getattr(self.local_instance, name)

            def launch(self):
                """
                Universal entry point for running a cls instance in distributed mode
                """
                with proxy:
                    proxy.join()
                    for msg in proxy.receive():
                        self.handler_dict[msg.type](msg)

        return Wrapper

    return dist_decorator
