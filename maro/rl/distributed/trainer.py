# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pickle
from collections import deque

import zmq
from zmq.eventloop.ioloop import IOLoop
from zmq.eventloop.zmqstream import ZMQStream


def trainer(id_: str):
    socket = zmq.Context().socket(zmq.REQ)
    socket.setsockopt_string(zmq.IDENTITY, f"Trainer_{id_}")
    socket.connect("tcp://127.0.0.1:50001")
    socket = ZMQStream(socket)
    socket.send(b"READY")  # register to the dispatcher

    def train(sock, msg):
        client, _, request = msg
        request = pickle.loads(request)
        info = request["agent"].step(*request["args"], **request["kwargs"])
        request.update({"model": request["agent"].dump_model(), "info": info})
        del request["agent"]
        del request["args"]
        del request["kwargs"]
        sock.send_multipart([client, b"", pickle.dumps(request)])
    
    socket.on_recv_stream(train)
    IOLoop.instance().start()
