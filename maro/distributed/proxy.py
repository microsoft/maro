# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# native lib
import socket
import json
import time
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
from copy import deepcopy

# third party lib
import redis
import zmq

# private lib
from maro.distributed.message import Message
from examples.ecr.q_learning.distributed_mode.message_type import MsgType, PayloadKey, MsgStatus
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
        shared_memory_manager = multiprocessing.Manager()
        self._message_cache = shared_memory_manager.dict()

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

    def scatter(self, message_type: MsgType, destination_payload_list: list, multithread = False, multiprocess = False):
        """separate data, and send to peer"""
        receive_list = self._group_send(message_type, destination_payload_list)

        return self.get(receive_list)

    def iscatter(self, message_type: MsgType, destination_payload_list: list, multithread = False, multiprocess = False):
        return self._group_send(message_type, destination_payload_list)

    def broadcast(self, message_type: MsgType, destination: list, payload, multithread = False, multiprocess = False):
        """send data to all peers"""
        destination_payload_list = [(dest, payload) for dest in destination]

        receive_list = self._group_send(message_type, destination_payload_list)

        return self.get(receive_list)

    def ibroadcast(self, message_type: MsgType, destination: list, payload, multithread = False, multiprocess = False):
        destination_payload_list = [(dest, payload) for dest in destination]

        return self._group_send(message_type, destination_payload_list)

    def send(self, message: Message):
        """Send a message to a remote peer

        Args:
            message: message to be sent
        """
        self._message_cache[message.message_id] = MsgStatus.SEND_MESSAGE

        if not hasattr(self, '_send_channel'):
            raise Exception('No message recipient found. Are you using the right configuration?')

        source, destination = message.source, message.destination
        if message.destination not in self._send_channel:
            raise Exception(f"Recipient {destination} is not found in {source}'s peers. "
                            f"Are you using the right configuration?")
        self._send_channel[destination].send_pyobj(message)
        self._logger.debug(f'sent a {message.type} message to {message.destination}')
        self._message_cache[message.message_id] = MsgStatus.WAIT_MESSAGE

    def _group_send(self, message_type: MsgType, destination_payload_list: list, multithread = False, multiprocess = False):
        """ """
        message_id_list = []

        if multithread:
            executor = ThreadPoolExecutor(max_workers=len(destination_payload_list))
        if multiprocess:
            executor = ProcessPoolExecutor(max_workers=len(destination_payload_list))

        for destination, payload in destination_payload_list:
            single_message = Message(type=message_type, source=self._name,
                                    destination=destination,
                                    payload=payload)
            if single_message.destination not in self._send_channel:
                raise Exception(f"Recipient {destination} is not found in {self._name}'s peers. "
                                f"Are you using the right configuration?")
            if multithread or multiprocess:
                executor.submit(self.send, single_message)
            else:
                self.send(single_message)
            message_id_list.append(single_message.message_id)
            self._logger.debug(f'sent a {single_message.type} message to {single_message.destination}')

        if multithread or multiprocess:
            executor.shutdown()
        return message_id_list

    def get(self, receive_list):
        pending_receive_list = receive_list[:]
        msg_holder = []

        for msg_id in receive_list:
            if msg_id in list(self._message_cache.keys()):
                pending_receive_list.remove(msg_id)
                msg_holder.append(self._message_cache[msg_id])
                del self._message_cache[msg_id]
            
            if not pending_receive_list:
                return msg_holder
        
        for msg in self.receive():
            if msg.message_id in pending_receive_list:
                pending_receive_list.remove(msg.message_id)
                msg_holder.append(msg)
            else:
                self._message_cache[msg.message_id] = msg

            if not pending_receive_list:
                break

        return msg_holder
