# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import partial
from typing import Callable

# private lib
from maro.communication.proxy import Proxy
from maro.communication.registry_table import RegisterTable
        

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
            as the universal entry point for running a cls instance in distributed mode.
            """
            def __init__(self, *args, **kwargs):
                self.local_instance = cls(*args, **kwargs)
                self.proxy = proxy
                self._handler_function = {}
                self._registry_table = RegisterTable(self.proxy.get_peers())
                # use functools.partial to freeze handling function's local_instance and proxy
                # arguments to self.local_instance and self.proxy
                for handler_fn, constraint in handler_dict.items():
                    self._handler_function[handler_fn] = partial(handler_fn, self.local_instance, self.proxy)
                    self._registry_table.register_event_handler(constraint, handler_fn)

            def __getattr__(self, name):
                if name in self.__dict__:
                    return self.__dict__[name]

                return getattr(self.local_instance, name)

            def launch(self):
                """
                Universal entry point for running a cls instance in distributed mode.
                """
                for msg in self.proxy.receive():
                    self._registry_table.push(msg)
                    triggered_event = self._registry_table.get()

                    for handler_fn, msg_lst in triggered_event:
                        self._handler_function[handler_fn](msg_lst)
                                
        return Wrapper

    return dist_decorator
