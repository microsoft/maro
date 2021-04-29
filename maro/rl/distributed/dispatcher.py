# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import deque

import zmq
from zmq.eventloop.ioloop import IOLoop
from zmq.eventloop.zmqstream import ZMQStream


class LRUQueue(object):
    """LRUQueue class using ZMQStream/IOLoop for event dispatching.

    Code adapted from https://zguide.zeromq.org/docs/chapter3/#A-High-Level-API-for-ZeroMQ.
    """
    def __init__(self):
        self._context = zmq.Context.instance()
        frontend = self._context.socket(zmq.ROUTER)
        frontend.bind("tcp://127.0.0.1:50000")
        backend = self._context.socket(zmq.ROUTER)
        backend.bind("tcp://127.0.0.1:50001")
        self._workers = deque()

        self._frontend = ZMQStream(frontend)
        self._backend = ZMQStream(backend)
        self._backend.on_recv(self._handle_backend)

    def _handle_backend(self, msg):
        # Queue worker ID for LRU routing
        worker, empty, client = msg[:3]
        # add worker back to the list of workers
        self._workers.append(worker)
        assert empty == b""
        # Third frame is READY or else a client reply address
        # If client reply, send rest back to frontend
        if client != b"READY":
            empty, reply = msg[3:]
            assert empty == b""
            self._frontend.send_multipart([client, b'', reply])

        # Start accepting frontend messages now that at least one worker is free.
        self._frontend.on_recv(self._handle_frontend)

    def _handle_frontend(self, msg):
        # Now get next client request, route to LRU worker
        # Client request is [address][empty][request]
        client, empty, request = msg
        assert empty == b""
        #  Dequeue and drop the next worker address
        self._backend.send_multipart([self._workers.popleft(), b'', client, b'', request])
        if not self._workers:
            # stop receiving until workers become available again
            self._frontend.stop_on_recv()


def start_dispatcher():
    dispatcher = LRUQueue()
    IOLoop.instance().start()
