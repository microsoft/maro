# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# native lib
import socket
import json
import time
import logging
from copy import deepcopy

# third party lib
import redis
import zmq

# private lib
from maro.distributed.message import Message
from typing import List


class Proxy:
    """Communication module responsible for receiving and sending messages

    Args:
        group_name (str): identifier for the group of all distributed components
        component_name (str): unique identifier in the current group
        peer_name_list (list): list of recipients of messages sent by this component
        protocol (str): underlying transport-layer protocol for transferring messages
        redis_address (tuple): hostname and port of the Redis server
        max_retries (int): maximum number of retries before raising an exception
        retry_interval: int = 5
        logger: logger instance
    """
    def __init__(self, group_name, component_name, peer_name_list: list = None,
                 protocol='tcp', redis_address=('localhost', 6379), max_retries: int = 5,
                 retry_interval: int = 5, logger=None, msg_request=None):
        self._group_name = group_name
        self._name = component_name
        self._peer_name_list = peer_name_list
        self._protocol = protocol
        self._redis_address = redis_address
        self._max_retries = max_retries
        self._retry_interval = retry_interval
        self._ip_address = socket.gethostbyname(socket.gethostname())
        self._logger = logger if logger else logging
        self._msg_request = [{'request': request,
                              'remain': request, 
                              'msg_list': []} for request in msg_request]

    @property
    def group_name(self) -> str:
        """str: group name"""
        return self._group_name

    @property
    def name(self) -> str:
        """str: component name"""
        return self._name

    @property
    def peers(self) -> List[str]:
        """List[str]: list of message receiver name"""
        return self._peer_name_list[:]

    def __del__(self):
        self._redis_connection.hdel(self._group_name, self._name)

    def join(self):
        """Join the communication network for the experiment given by experiment_name with ID given by name.
        Specifically, it creates sockets for receiving (pulling) messages from its ZMQ peers and uploads
        the receiving address to the Redis server. It then attempts to connect to remote peers by querying
        the Redis server for their addresses
        """
        self._zmq_context = zmq.Context()
        self._redis_connection = redis.StrictRedis(host=self._redis_address[0], port=self._redis_address[1])
        self._set_up_receiving()
        if self._peer_name_list is not None:
            self._connect_to_peers()

    def _set_up_receiving(self):
        # create a receiving socket, bind it to a random port and upload the address info to the Redis server
        self._receiver = self._zmq_context.socket(zmq.PULL)
        recv_port = self._receiver.bind_to_random_port(f'{self._protocol}://*')
        recv_address = [self._ip_address, recv_port]
        self._redis_connection.hset(self._group_name, self._name, json.dumps(recv_address))
        self._logger.info(f'{self._name} set to receive messages at {self._ip_address}:{recv_port}')

    def _connect_to_peers(self):
        # create send_channel attribute and initialize it to an empty dict
        peer_address_dict, self._send_channel = {}, {}
        for peer_name in self._peer_name_list:
            retried, connected = 0, False
            while retried < self._max_retries:
                try:
                    ip, port = json.loads(self._redis_connection.hget(self._group_name, peer_name))
                    remote_address = f'{self._protocol}://{ip}:{port}'
                    self._send_channel[peer_name] = self._zmq_context.socket(zmq.PUSH)
                    self._send_channel[peer_name].connect(remote_address)
                    peer_address_dict[peer_name] = remote_address
                    connected = True
                    break
                except:
                    self._logger.error(f'Failed to connect to {peer_name}. Retrying in {self._retry_interval} seconds')
                    time.sleep(self._retry_interval)
                    retried += 1

            if not connected:
                raise ConnectionAbortedError(f'Cannot connect to {peer_name}. Please check your configurations. ')

        self._logger.info(f'{self._name} set to send messages to {peer_address_dict}')

    def receive_once(self):
        """Receive one message from ZMQ"""
        return self._receiver.recv_pyobj()

    def receive(self):
        """Receive messages from ZMQ"""
        while True:
            yield self._receiver.recv_pyobj()

    def _msg_decompose(self, message):
        """return [payload, source, type, destination, required] """
        return message.payload, message.source, message.type, message.destination, message.operation

    def scatter(self, message):
        """separate data, and send to peer"""
        payload, msg_source, msg_type, msg_dest, msg_operation = self._msg_decompose(message)
        
        for idx, pld in enumerate(payload):
            destination = msg_dest[idx]
            scatter_message = Message(type=msg_type, source=msg_source,
                                    destination=destination,
                                    payload=pld,
                                    operation=msg_operation)
            self.send(scatter_message)

    def boardcast(self, message):
        """send data to all peers"""
        payload, msg_source, msg_type, msg_dest, msg_operation = self._msg_decompose(message)

        for destination in msg_dest:
            boardcast_message = Message(type=msg_type, source=msg_source,
                                    destination=destination,
                                    payload=payload, 
                                    operation=msg_operation)
            self.send(boardcast_message)

    def send(self, message: Message):
        """Send a message to a remote peer

        Args:
            message: message to be sent
        """
        if not hasattr(self, '_send_channel'):
            raise Exception('No message recipient found. Are you using the right configuration?')

        source, destination = message.source, message.destination
        if message.destination not in self._send_channel:
            raise Exception(f"Recipient {destination} is not found in {source}'s peers. "
                            f"Are you using the right configuration?")
        self._send_channel[destination].send_pyobj(message)
        self._logger.debug(f'sent a {message.type} message to {message.destination}')

    def msg_handler(self, message: Message):
        for req_dict in self._msg_request:
            for key, value in req_dict['remain'].items():
                if key == (message.source, message.type):
                    if value > 1:
                        req_dict['remain'].update({key: value - 1})
                    else:
                        del req_dict['remain'][key]
                    req_dict['msg_list'].append(message)})

                    if not req_dict['remain']:
                        request_msg_list = req_dict['msg_lst'][:]
                        req_dict['msg_lst'] = []
                        req_dict['remain'] = req_dict['request']
 
                        return request_msg_list
                    
                    return None

        return list(message)


            
