# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


# native lib
import socket
import json
import pickle
import time
import logging
import re

# third party lib
import redis
import zmq

# private lib
from maro.distributed.message import Message
from typing import List


class Proxy:
    """Communication module responsible for receiving and sending messages

    Args:
        receive_enabled (bool): flag indicating whether the component receives messages
        audience_list (list): list of recipients of messages sent by this component
        protocol (str): underlying transport-layer protocol for transferring messages
        redis_host (str): host name of the Redis server
        redis_port (int): port number of the Redis Server
        max_retries (int): maximum number of retries before raising an exception
        retry_interval: int = 5
        logger: logger instance
    """
    def __init__(self, receive_enabled: bool = False, audience_list: list = None, protocol='tcp',
                 redis_host='localhost', redis_port=6379, max_retries: int = 5, retry_interval: int = 5,
                 logger=None):
        self._name = None
        self._receive_enabled = receive_enabled
        self._audience_list = audience_list
        self._protocol = protocol
        self._redis_host = redis_host
        self._redis_port = redis_port
        self._max_retries = max_retries
        self._retry_interval = retry_interval
        self._ip_address = socket.gethostbyname(socket.gethostname())
        self._logger = logger if logger else logging

    @property
    def audience(self) -> List[str]:
        """List[str]: list of message receiver name"""
        return self._audience_list[:]

    def join(self, group_name: str, component_name: str):
        """Join the communication network for the experiment given by experiment_name with ID given by name.
        Specifically, it creates sockets for receiving (pulling) messages from its ZMQ peers and uploads
        the receiving address to the Redis server. It then attempts to connect to remote peers by querying
        the Redis server for their addresses

        Args:
            group_name (str): identifier for the group of all distributed components
            component_name (str): unique identifier in the current group
        """
        self._zmq_context = zmq.Context()
        self._redis_connection = redis.StrictRedis(host=self._redis_host, port=self._redis_port)
        self._name = component_name
        if self._receive_enabled:
            recv_port = self._create_receiver()
            self._logger.info(f'{self._name} set to receive messages at {self._ip_address}:{recv_port}')
            self._register_to_redis(group_name, component_name, recv_port)
        if self._audience_list is not None:
            peers = self._connect_to_peers(group_name)
            self._logger.info(f'{component_name} set to send messages to {peers}')

    def _create_receiver(self):
        # create a receiving socket, bind it to a random port and upload the address info to the Redis server
        self._receiver = self._zmq_context.socket(zmq.PULL)
        return self._receiver.bind_to_random_port(f'{self._protocol}://*')

    def _register_to_redis(self, group_name, component_name, recv_port):
        if self._redis_connection.hexists(group_name, component_name):
            raise Exception(f'Record for {component_name} found under {group_name}. '
                            f'To avoid unwanted overwriting, please use a new experiment name')
        recv_address = [self._ip_address, recv_port]
        self._redis_connection.hset(group_name, component_name, json.dumps(recv_address))

    def _connect_to_peers(self, group_name):
        # create send_channel attribute and initialize it to an empty dict
        peer_address_dict, self._send_channel = {}, {}
        for peer_name in self._audience_list:
            retried, connected = 0, False
            while retried < self._max_retries:
                try:
                    ip, port = json.loads(self._redis_connection.hget(group_name, peer_name))
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

        return peer_address_dict

    def receive(self):
        """Receive messages from ZMQ"""
        if not hasattr(self, '_receiver'):
            raise Exception('Message receiving is not enabled for this component. '
                            'Are you using the right configuration?')
        while True:
            yield Message.recv(self._receiver)

    def send(self, peer_name: str, msg_type: object, msg_body: str):
        """Send a message to a remote peer

        Args:
            peer_name (str): remote peer ID
            msg_type (object): message type
            msg_body (str): message body
        """
        if not hasattr(self, '_send_channel'):
            raise Exception('No message recipient found. Are you using the right configuration?')
        self._send_channel[peer_name].send(self._prepare_msg(peer_name, msg_type, msg_body))
        self._logger.debug(f'sent a {msg_type} message to {peer_name}')

    def multicast(self, match: str, msg_type: object, msg_body: str):
        """Send messages to all remote peers whose name matches a specified pattern. For instance,
        multicast('*experience*', ...) will send to all components whose name contains "experience"

        Args:
            match (str): specifies regex pattern for peer name. All peers whose name matches the pattern
                   will be the target of the message, i.e.,
            msg_type (object): message type
            msg_body (str): message body
        """
        if not hasattr(self, '_send_channel'):
            raise Exception('No message recipient found. Are you using the right configuration?')
        pattern = re.compile(match)
        for peer_name, chn in self._send_channel.items():
            if pattern.search(peer_name):
                chn.send(self._prepare_msg(peer_name, msg_type, msg_body))
                self._logger.debug(f'sent a {msg_type} message to {peer_name}')

    def _prepare_msg(self, peer_name, msg_type, msg_body):
        """Constructs a message with a fixed-length header to indicate message body length in bytes"""
        return pickle.dumps({'src': self._name, 'dest': peer_name, 'type': msg_type, 'body': msg_body})
