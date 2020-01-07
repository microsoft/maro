# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


class Message(object):
    """Message object used to hold information between receiver and sender"""
    def __init__(self, type, source, destination, body=None):
        self.type = type
        self.source = source
        self.destination = destination
        self.body = {} if body is None else body
