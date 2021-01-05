# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# native lib
from functools import partial
from typing import Callable

# private lib
from .proxy import Proxy
from .registry_table import RegisterTable


def dist(proxy: Proxy, handler_dict: {object: Callable}):
    """
    A decorator used to inject a ``communication module`` and ``message handlers``
    to a local class so that it can be run in distributed mode.
    """
    def dist_decorator(cls):
        class Wrapper:
            """A wrapper class for ``cls``, the class to be decorated.

            It contains a reference to the ``proxy`` and a ``message handler`` lookup table and defines a launch method
            as the universal entry point for running a ``cls`` instance in distributed mode.
            """
            def __init__(self, *args, **kwargs):
                self.local_instance = cls(*args, **kwargs)
                self.proxy = proxy
                self._handler_function = {}
                self._registry_table = RegisterTable(self.proxy.peers_name)
                # Use functools.partial to freeze handling function's local_instance and proxy
                # arguments to self.local_instance and self.proxy.
                for constraint, handler_fun in handler_dict.items():
                    self._handler_function[handler_fun] = partial(handler_fun, self.local_instance, self.proxy)
                    self._registry_table.register_event_handler(constraint, handler_fun)

            def __getattr__(self, name):
                if name in self.__dict__:
                    return self.__dict__[name]

                return getattr(self.local_instance, name)

            def launch(self):
                """Universal entry point for running a ``cls`` instance in distributed mode."""
                for message in self.proxy.receive():
                    self._registry_table.push(message)
                    triggered_event = self._registry_table.get()

                    for handler_fun, message_list in triggered_event:
                        self._handler_function[handler_fun](message_list)

        return Wrapper

    return dist_decorator
