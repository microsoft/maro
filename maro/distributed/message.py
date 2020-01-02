# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import pickle


class Message(object):
    """Message object used to hold information between receiver and sender"""
    def __init__(self, type_, src, dest, body=None):
        self.type = type_
        self.src = src
        self.dest = dest
        self.body = {} if body is None else body

    @classmethod
    def recv(cls, sock) -> object:
        """object: Reads key-value message from socket, returns new Message instance."""
        msg = pickle.loads(sock.recv())
        return cls(msg['type'], msg['src'], msg['dest'], msg['body'])
