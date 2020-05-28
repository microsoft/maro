# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# private lib
from maro.distributed.proxy import Proxy
from typing import Callable
from functools import partial
from copy import deepcopy
from maro.distributed.message import Message, HandlerKey, ConstraintType


class HandlerTrigger():
    def __init__(self, handler_dict):
        self._msg_request_handle_list = []
        for info in handler_dict:
            # CONSTRAINT class
            # remain -> count
            self._msg_request_handle_list.append({HandlerKey.CONSTRAINT: info[HandlerKey.CONSTRAINT],
                                                  HandlerKey.REMAIN: deepcopy(info[HandlerKey.CONSTRAINT]), 
                                                  HandlerKey.MSG_LIST: [],
                                                  HandlerKey.HANDLER_FN: info[HandlerKey.HANDLER_FN]})
        self._message_cache = []

    def _message_trigger(self, message):
        message_spend = False

        for req_dict in self._msg_request_handle_list:
            for constraint, num in list(req_dict[HandlerKey.REMAIN].items()):
                component_name = message.source[:message.source.rindex("_")]
                if constraint == (message.source, message.type) or constraint == (ConstraintType.ANY_SOURCE, message.type) or \
                   constraint == (message.source, ConstraintType.ANY_TYPE) or constraint == (component_name, message.type):
                    if num > 1:
                        req_dict[HandlerKey.REMAIN].update({constraint: num - 1})
                    else:
                        del req_dict[HandlerKey.REMAIN][constraint]
                
                    req_dict[HandlerKey.MSG_LIST].append(message)
                    message_spend = True

                if not req_dict[HandlerKey.REMAIN]:
                    request_msg_list = req_dict[HandlerKey.MSG_LIST][:]
                    req_dict[HandlerKey.MSG_LIST] = []
                    req_dict[HandlerKey.REMAIN] = deepcopy(req_dict[HandlerKey.CONSTRAINT])
                    self._satisfied_constraint.append((req_dict[HandlerKey.HANDLER_FN], request_msg_list))
        
        if not message_spend:
            return message

    def message_trigger(self, message):
        self._message_cache.append(message)
        self._satisfied_constraint=[]
        unused_message_list = []
        
        for msg in self._message_cache:
            ret_msg = self._message_trigger(msg)
            if ret_msg:
                unused_message_list.append(ret_msg)
        
        self._message_cache = unused_message_list[:]

        return self._satisfied_constraint
        

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
                self._handler_function = {}
                # use functools.partial to freeze handling function's local_instance and proxy
                # arguments to self.local_instance and self.proxy
                for info in handler_dict:
                    self._handler_function[info[HandlerKey.HANDLER_FN]] = partial(info[HandlerKey.HANDLER_FN], self.local_instance, self.proxy)
                self._handler_trigger = HandlerTrigger(handler_dict)

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
                    satisfied_constraint = self._handler_trigger.message_trigger(msg)  

                    for handler_fn, msg_lst in satisfied_constraint:
                        self._handler_function[handler_fn](msg_lst)         
                                
        return Wrapper

    return dist_decorator
