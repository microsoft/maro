# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

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
