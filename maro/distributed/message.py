# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from enum import Enum

initial_message_id = 0

class Message(object):
    """Message object used to hold information between receiver and sender"""
    def __init__(self, type, source, destination, payload=None, message_id=None):
        self.type = type
        self.source = source
        self.destination = destination
        self.payload = {} if payload is None else payload
        if message_id is None:
            global initial_message_id
            self.message_id = initial_message_id
            initial_message_id += 1
        else:
            self.message_id = message_id


class MsgStatus(Enum):
    SEND_MESSAGE = 0
    WAIT_MESSAGE = 1


class SocketType(Enum):
    ZMQ_PUB = 0
    ZMQ_PUSH = 1


class HandlerKey(Enum):
    CONSTRAINT = 0
    HANDLER_FN = 1
    REMAIN = 2
    MSG_LIST = 3


class ConstraintType(Enum):
    ANY_SOURCE = 0
    ANY_TYPE = 0 
