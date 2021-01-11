# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# native lib
import pickle
import socket
import sys
from typing import Dict

# third party package
import zmq

# private package
from maro.utils import DummyLogger
from maro.utils.exception.communication_exception import (
    DriverReceiveError, DriverSendError, PeersConnectionError, PeersDisconnectionError, PendingToSend, SocketTypeError
)
from maro.utils.exit_code import NON_RESTART_EXIT_CODE

from ..message import Message
from ..utils import default_parameters
from .abs_driver import AbsDriver

PROTOCOL = default_parameters.driver.zmq.protocol
SEND_TIMEOUT = default_parameters.driver.zmq.send_timeout
RECEIVE_TIMEOUT = default_parameters.driver.zmq.receive_timeout


class ZmqDriver(AbsDriver):
    """The communication driver based on ``ZMQ``.

    Args:
        component_type (str): Component type in the current group.
        protocol (str): The underlying transport-layer protocol for transferring messages. Defaults to tcp.
        send_timeout (int): The timeout in milliseconds for sending message. If -1, no timeout (infinite).
            Defaults to -1.
        receive_timeout (int): The timeout in milliseconds for receiving message. If -1, no timeout (infinite).
            Defaults to -1.
        logger: The logger instance or DummyLogger. Defaults to DummyLogger().
    """

    def __init__(
        self,
        component_type: str,
        protocol: str = PROTOCOL,
        send_timeout: int = SEND_TIMEOUT,
        receive_timeout: int = RECEIVE_TIMEOUT,
        logger=DummyLogger()
    ):
        self._component_type = component_type
        self._protocol = protocol
        self._send_timeout = send_timeout
        self._receive_timeout = receive_timeout
        self._ip_address = socket.gethostbyname(socket.gethostname())
        self._zmq_context = zmq.Context()
        self._disconnected_peer_name_list = []
        self._logger = logger

        self._setup_sockets()

    def _setup_sockets(self):
        """Setup three kinds of sockets, and one poller.

        - ``unicast_receiver``: The ``zmq.PULL`` socket, use for receiving message from one-to-one communication,
        - ``broadcast_sender``: The ``zmq.PUB`` socket, use for broadcasting message to all subscribers,
        - ``broadcast_receiver``: The ``zmq.SUB`` socket, use for listening message from broadcast.
        - ``poller``: The zmq output multiplexing, use for receiving message from ``zmq.PULL`` socket and \
            ``zmq.SUB`` socket.
        """
        self._unicast_receiver = self._zmq_context.socket(zmq.PULL)
        unicast_receiver_port = self._unicast_receiver.bind_to_random_port(f"{self._protocol}://*")
        self._logger.info(f"Receive message via unicasting at {self._ip_address}:{unicast_receiver_port}.")

        # Dict about zmq.PUSH sockets, fulfills in self.connect.
        self._unicast_sender_dict = {}

        self._broadcast_sender = self._zmq_context.socket(zmq.PUB)
        self._broadcast_sender.setsockopt(zmq.SNDTIMEO, self._send_timeout)

        self._broadcast_receiver = self._zmq_context.socket(zmq.SUB)
        self._broadcast_receiver.setsockopt(zmq.SUBSCRIBE, self._component_type.encode())
        broadcast_receiver_port = self._broadcast_receiver.bind_to_random_port(f"{self._protocol}://*")
        self._logger.info(f"Subscriber message at {self._ip_address}:{broadcast_receiver_port}.")

        # Record own sockets' address.
        self._address = {
            zmq.PULL: f"{self._protocol}://{self._ip_address}:{unicast_receiver_port}",
            zmq.SUB: f"{self._protocol}://{self._ip_address}:{broadcast_receiver_port}"
        }

        self._poller = zmq.Poller()
        self._poller.register(self._unicast_receiver, zmq.POLLIN)
        self._poller.register(self._broadcast_receiver, zmq.POLLIN)

    @property
    def address(self) -> Dict[int, str]:
        """
        Returns:
            Dict[int, str]: The sockets' address Dict of ``zmq.PULL`` socket and ``zmq.SUB`` socket.
            The key of dict is the socket's type, while the value of dict is socket's ip address,
            which forms by protocol+ip+port.

        Example:
            Dict{zmq.PULL: "tcp://0.0.0.0:1234", zmq.SUB: "tcp://0.0.0.0:1235"}
        """
        return self._address

    def connect(self, peers_address_dict: Dict[str, Dict[str, str]]):
        """Build a connection with all peers in peers socket address.

        Set up the unicast sender which is ``zmq.PUSH`` socket and the broadcast sender which is ``zmq.PUB`` socket.

        Args:
            peers_address_dict (Dict[str, Dict[str, str]]): Peers' socket address dict.
                The key of dict is the peer's name, while the value of dict is the peer's socket connection address.
                E.g. Dict{'peer1', Dict[zmq.PULL, 'tcp://0.0.0.0:1234']}.
        """
        for peer_name, address_dict in peers_address_dict.items():
            for socket_type, address in address_dict.items():
                try:
                    if int(socket_type) == zmq.PULL:
                        self._unicast_sender_dict[peer_name] = self._zmq_context.socket(zmq.PUSH)
                        self._unicast_sender_dict[peer_name].setsockopt(zmq.SNDTIMEO, self._send_timeout)
                        self._unicast_sender_dict[peer_name].connect(address)
                        self._logger.info(f"Connects to {peer_name} via unicasting.")
                    elif int(socket_type) == zmq.SUB:
                        self._broadcast_sender.connect(address)
                        self._logger.info(f"Connects to {peer_name} via broadcasting.")
                    else:
                        raise SocketTypeError(f"Unrecognized socket type {socket_type}.")
                except Exception as e:
                    raise PeersConnectionError(f"Driver cannot connect to {peer_name}! Due to {str(e)}")

            if peer_name in self._disconnected_peer_name_list:
                self._disconnected_peer_name_list.remove(peer_name)

    def disconnect(self, peers_address_dict: Dict[str, Dict[str, str]]):
        """Disconnect with all peers in peers socket address.

        Disconnect and delete the unicast sender which is ``zmq.PUSH`` socket for the peers in dict.

        Args:
            peers_address_dict (Dict[str, Dict[str, str]]): Peers' socket address dict.
                The key of dict is the peer's name, while the value of dict is the peer's socket connection address.
                E.g. Dict{'peer1', Dict[zmq.PULL, 'tcp://0.0.0.0:1234']}.
        """
        for peer_name, address_dict in peers_address_dict.items():
            for socket_type, address in address_dict.items():
                try:
                    if int(socket_type) == zmq.PULL:
                        self._unicast_sender_dict[peer_name].disconnect(address)
                        del self._unicast_sender_dict[peer_name]
                    elif int(socket_type) == zmq.SUB:
                        self._broadcast_sender.disconnect(address)
                    else:
                        raise SocketTypeError(f"Unrecognized socket type {socket_type}.")
                except Exception as e:
                    raise PeersDisconnectionError(f"Driver cannot disconnect to {peer_name}! Due to {str(e)}")

            self._disconnected_peer_name_list.append(peer_name)
            self._logger.info(f"Disconnected with {peer_name}.")

    def receive(self, is_continuous: bool = True, timeout: int = None):
        """Receive message from ``zmq.POLLER``.

        Args:
            is_continuous (bool): Continuously receive message or not. Defaults to True.

        Yields:
            recv_message (Message): The received message from the poller.
        """
        while True:
            receive_timeout = timeout if timeout else self._receive_timeout
            try:
                sockets = dict(self._poller.poll(receive_timeout))
            except Exception as e:
                raise DriverReceiveError(f"Driver cannot receive message as {e}")

            if self._unicast_receiver in sockets:
                recv_message = self._unicast_receiver.recv_pyobj()
                self._logger.debug(f"Receive a message from {recv_message.source} through unicast receiver.")
            elif self._broadcast_receiver in sockets:
                _, recv_message = self._broadcast_receiver.recv_multipart()
                recv_message = pickle.loads(recv_message)
                self._logger.debug(f"Receive a message from {recv_message.source} through broadcast receiver.")
            else:
                self._logger.debug(f"Cannot receive any message within {receive_timeout}.")
                return

            yield recv_message

            if not is_continuous:
                break

    def send(self, message: Message):
        """Send message.

        Args:
            message (class): Message to be sent.
        """
        try:
            self._unicast_sender_dict[message.destination].send_pyobj(message)
            self._logger.debug(f"Send a {message.tag} message to {message.destination}.")
        except KeyError as key_error:
            if message.destination in self._disconnected_peer_name_list:
                raise PendingToSend(f"Temporary failure to send message to {message.destination}, may rejoin later.")
            else:
                self._logger.error(f"Failure to send message caused by: {key_error}")
                sys.exit(NON_RESTART_EXIT_CODE)
        except Exception as e:
            raise DriverSendError(f"Failure to send message caused by: {e}")

    def broadcast(self, topic: str, message: Message):
        """Broadcast message.

        Args:
            topic(str): The topic of broadcast.
            message(class): Message to be sent.
        """
        try:
            self._broadcast_sender.send_multipart([topic.encode(), pickle.dumps(message)])
            self._logger.debug(f"Broadcast a {message.tag} message to all {topic}.")
        except Exception as e:
            raise DriverSendError(f"Failure to broadcast message caused by: {e}")

    def close(self):
        """Close ZMQ context and sockets."""
        # Avoid hanging infinitely
        self._zmq_context.setsockopt(zmq.LINGER, 0)

        # Close all sockets
        self._broadcast_receiver.close()
        self._broadcast_sender.close()
        self._unicast_receiver.close()
        for unicast_sender in self._unicast_sender_dict.values():
            unicast_sender.close()

        self._zmq_context.term()
