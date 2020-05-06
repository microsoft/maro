# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


class Message(object):
    """Message object used to hold information between receiver and sender"""
    def __init__(self, type, source, destination, payload=None, mid=None):
        self.type = type
        self.source = source
        self.destination = destination
        self.payload = {} if payload is None else payload
        self.mid = mid
        # self.required = {} if required is None else required
        # required example: {'request': {(env1, msg_type): #num, (env2, msg_type): #num}, 'operation': add/append/max/min}

    def set_mid(self, mid):
        self.mid = mid
