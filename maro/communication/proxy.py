# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# native lib
import itertools
import json
import signal
import sys
import time
import uuid
from collections import defaultdict, namedtuple
from enum import Enum
from typing import Dict, List, Union

# third party lib
import redis

# private lib
from maro.utils import DummyLogger, InternalLogger
from maro.utils.exception.communication_exception import (
    DriverTypeError, InformationUncompletedError, PeersMissError, RedisConnectionError
)

from .driver import DriverType, ZmqDriver
from .message import Message, NotificationSessionStage, SessionMessage, SessionType, TaskSessionStage
from .utils import default_parameters

_PEER_INFO = namedtuple("PEER_INFO", ["hash_table_name", "expected_number"])
HOST = default_parameters.proxy.redis.host
PORT = default_parameters.proxy.redis.port
MAX_RETRIES = default_parameters.proxy.redis.max_retries
BASE_RETRY_INTERVAL = default_parameters.proxy.redis.base_retry_interval
FAULT_TOLERANT = default_parameters.proxy.fault_tolerant
DELAY_FOR_SLOW_JOINER = default_parameters.proxy.delay_for_slow_joiner


class Proxy:
    """The communication module is responsible for receiving and sending messages.

    There are three ways of sending messages: ``send``, ``scatter``, and ``broadcast``. Also, there are two ways to
    receive messages from other peers: ``receive`` and ``receive_by_id``.

    Args:
        group_name (str): Identifier for the group of all distributed components.
        component_type (str): Component's type in the current group.
        expected_peers (Dict): Dict of peers' information which contains peer type and expected number.
            E.g. Dict['learner': 1, 'actor': 2]
        driver_type (Enum): A type of communication driver class uses to communicate with other components.
            Defaults to ``DriverType.ZMQ``.
        driver_parameters (Dict): The arguments for communication driver class initial. Defaults to None.
        redis_address (Tuple): Hostname and port of the Redis server. Defaults to ("localhost", 6379).
        max_retries (int): Maximum number of retries before raising an exception. Defaults to 5.
        base_retry_interval (float): The time interval between attempts. Defaults to 0.1.
        fault_tolerant (bool): Proxy can tolerate sending message error or not. Defaults to False.
        log_enable (bool): Open internal logger or not. Defaults to True.
    """

    def __init__(
        self, group_name: str, component_type: str, expected_peers: dict,
        driver_type: DriverType = DriverType.ZMQ, driver_parameters: dict = None,
        redis_address=(HOST, PORT), max_retries: int = MAX_RETRIES,
        base_retry_interval: float = BASE_RETRY_INTERVAL,
        fault_tolerant: bool = FAULT_TOLERANT, log_enable: bool = True
    ):
        self._group_name = group_name
        self._component_type = component_type
        self._redis_hash_name = f"{self._group_name}:{self._component_type}"
        unique_id = str(uuid.uuid1()).replace("-", "")
        self._name = f"{self._component_type}_proxy_{unique_id}"
        self._driver_type = driver_type
        self._driver_parameters = driver_parameters
        self._max_retries = max_retries
        self._retry_interval = base_retry_interval
        self._is_enable_fault_tolerant = fault_tolerant
        self._log_enable = log_enable
        self._logger = InternalLogger(component_name=self._name) if self._log_enable else DummyLogger()

        try:
            self._redis_connection = redis.Redis(host=redis_address[0], port=redis_address[1])
        except Exception as e:
            raise RedisConnectionError(f"{self._name} failure to connect to redis server due to {e}")

        # Record the peer's redis information.
        self._peers_info_dict = {}
        for peer_type, number in expected_peers.items():
            self._peers_info_dict[peer_type] = _PEER_INFO(
                hash_table_name=f"{self._group_name}:{peer_type}",
                expected_number=number
            )
        # Record connected peers' name.
        self._onboard_peers_name_dict = {}
        # Temporary store the message.
        self._message_cache = defaultdict(list)

        self._join()

    def _signal_handler(self, signum, frame):
        self._redis_connection.hdel(self._redis_hash_name, self._name)
        self._logger.critical(f"{self._name} received Signal: {signum} at frame: {frame}")
        sys.exit(signum)

    def _join(self):
        """Join the communication network for the experiment given by experiment_name with ID given by name.

        Specifically, it gets sockets' address for receiving (pulling) messages from its driver and uploads
        the receiving address to the Redis server. It then attempts to collect remote peers' receiving address
        by querying the Redis server. Finally, ask its driver to connect remote peers using those receiving address.
        """
        self._register_redis()
        self._get_peers_list()
        self._build_connection()
        # TODO: Handle slow joiner for PUB/SUB.
        time.sleep(DELAY_FOR_SLOW_JOINER)

    def __del__(self):
        self._redis_connection.hdel(self._redis_hash_name, self._name)

    def _register_redis(self):
        """Self-registration on Redis and driver initialization.

        Redis store structure:
        Hash Table: name: group name + peer's type,
                    table (Dict[]): The key of table is the peer's name,
                                    the value of table is the peer's socket address.
        """
        if self._driver_type == DriverType.ZMQ:
            self._driver = ZmqDriver(**self._driver_parameters, logger=self._logger) if self._driver_parameters else \
                ZmqDriver(logger=self._logger)
        else:
            raise DriverTypeError(f"Unsupported driver type {self._driver_type}, please use DriverType class.")

        driver_address = self._driver.address
        self._redis_connection.hset(self._redis_hash_name, self._name, json.dumps(driver_address))

        # Handle interrupt signal for clearing Redis record.
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except Exception as e:
            self._logger.critical(
                "Signal detector disable. This may cause dirty data to be left in the Redis! "
                "To avoid this, please use multiprocess or make sure it can exit successfully."
                f"Due to {str(e)}."
            )

    def _get_peers_list(self):
        """To collect all peers' name in the same group (group name) from Redis."""
        if not self._peers_info_dict:
            raise PeersMissError(f"Cannot get {self._name}\'s peers.")

        for peer_type in self._peers_info_dict.keys():
            peer_hash_name, peer_number = self._peers_info_dict[peer_type]
            retry_number = 0
            expected_peers_name = []
            while retry_number < self._max_retries:
                if self._redis_connection.hlen(peer_hash_name) >= peer_number:
                    expected_peers_name = self._redis_connection.hkeys(peer_hash_name)
                    expected_peers_name = [peer.decode() for peer in expected_peers_name]
                    if len(expected_peers_name) > peer_number:
                        expected_peers_name = expected_peers_name[:peer_number]
                    self._logger.debug(f"{self._name} successfully get all {peer_type}\'s name.")
                    break
                else:
                    self._logger.debug(
                        f"{self._name} failed to get {peer_type}\'s name. Retrying in "
                        f"{self._retry_interval * (2 ** retry_number)} seconds."
                    )
                    time.sleep(self._retry_interval * (2 ** retry_number))
                    retry_number += 1

            if not expected_peers_name:
                raise InformationUncompletedError(
                    f"{self._name} failure to get enough number of {peer_type} from redis.")

            self._onboard_peers_name_dict[peer_type] = expected_peers_name

    def _build_connection(self):
        """Grabbing all peers' address from Redis, and connect all peers in driver."""
        peers_socket_dict = {}
        for peer_type, name_list in self._onboard_peers_name_dict.items():
            try:
                peers_socket_value = self._redis_connection.hmget(
                    self._peers_info_dict[peer_type].hash_table_name,
                    name_list
                )
                for idx, peer_name in enumerate(name_list):
                    peers_socket_dict[peer_name] = json.loads(peers_socket_value[idx])
                    self._logger.debug(f"{self._name} successfully get {peer_name}\'s socket address")
            except Exception as e:
                raise InformationUncompletedError(f"{self._name} failed to get {name_list}\'s address. Due to {str(e)}")

        self._driver.connect(peers_socket_dict)

    @property
    def group_name(self) -> str:
        """str: Identifier for the group of all communication components."""
        return self._group_name

    @property
    def component_name(self) -> str:
        """str: Unique identifier in the current group."""
        return self._name

    @property
    def peers(self) -> Dict:
        """Dict: The ``Dict`` of all connected peers' names, stored by peer type."""
        return self._onboard_peers_name_dict

    def get_peers(self, component_type: str = "*") -> List[str]:
        """Return peers' name list depending on the component type.

        Args:
            component_type (str): The peers' type, if ``*``, return all peers' name in the proxy. Defaults to ``*``.

        Returns:
            List[str]: List of peers' name.
        """
        if component_type == "*":
            return list(itertools.chain.from_iterable(self._onboard_peers_name_dict.values()))

        if component_type not in self._onboard_peers_name_dict.keys():
            raise PeersMissError(f"{component_type} not in current peers list!")

        return self._onboard_peers_name_dict[component_type]

    def receive(self, is_continuous: bool = True):
        """Receive messages from communication driver.

        Args:
            is_continuous (bool): Continuously receive message or not. Defaults to True.
        """
        return self._driver.receive(is_continuous)

    def receive_by_id(self, session_id_list: list) -> List[Message]:
        """Receive target messages from communication driver.

        Args:
            session_id_list List[str]: List of ``session_id``.
                E.g. ['0_learner0_actor0', '1_learner1_actor1', ...].

        Returns:
            List[Message]: List of received messages.
        """
        pending_message_list = session_id_list[:]
        received_message = []

        # Check message cache for saved messages.
        for msg_key in session_id_list:
            if msg_key in list(self._message_cache.keys()):
                for msg in self._message_cache[msg_key]:
                    pending_message_list.remove(msg_key)
                    received_message.append(msg)
                del self._message_cache[msg_key]

        if not pending_message_list:
            return received_message

        # Wait for incoming messages.
        for msg in self._driver.receive():
            msg_key = msg.session_id

            if msg_key in pending_message_list:
                pending_message_list.remove(msg_key)
                received_message.append(msg)
            else:
                self._message_cache[msg_key].append(msg)

            if not pending_message_list:
                break

        return received_message

    def _scatter(
        self, tag: Union[str, Enum], session_type: SessionType, destination_payload_list: list, session_id: str = None
    ) -> List[str]:
        """Scatters a list of data to peers, and return list of session id."""
        session_id_list = []

        for destination, payload in destination_payload_list:
            message = SessionMessage(
                tag=tag,
                source=self._name,
                destination=destination,
                session_id=session_id,
                payload=payload,
                session_type=session_type
            )
            sending_status = self._driver.send(message)

            if not sending_status:
                session_id_list.append(message.session_id)
            elif sending_status and self._is_enable_fault_tolerant:
                self._logger.warn(
                    f"{self._name} failure to send message to {message.destination}, as {str(sending_status)}")
            else:
                raise sending_status

        return session_id_list

    def scatter(
        self, tag: Union[str, Enum], session_type: SessionType, destination_payload_list: list, session_id: str = None
    ) -> List[Message]:
        """Scatters a list of data to peers, and return replied messages.

        Args:
            tag (str|Enum): Message's tag.
            session_type (Enum): Message's session type.
            destination_payload_list ([Tuple(str, object)]): The destination-payload list.
                The first item of the tuple in list is the message destination,
                and the second item of the tuple in list is the message payload.
            session_id (str): Message's session id. Defaults to None.

        Returns:
            List[Message]: List of replied message.
        """
        return self.receive_by_id(self._scatter(tag, session_type, destination_payload_list, session_id))

    def iscatter(
        self, tag: Union[str, Enum], session_type: SessionType, destination_payload_list: list, session_id: str = None
    ) -> List[str]:
        """Scatters a list of data to peers, and return list of message id.

        Args:
            tag (str|Enum): Message's tag.
            session_type (Enum): Message's session type.
            destination_payload_list ([Tuple(str, object)]): The destination-payload list.
                The first item of the tuple in list is the message's destination,
                and the second item of the tuple in list is the message's payload.
            session_id (str): Message's session id. Defaults to None.

        Returns:
            List[str]: List of message's session id.
        """
        return self._scatter(tag, session_type, destination_payload_list, session_id)

    def _broadcast(
        self, tag: Union[str, Enum], session_type: SessionType, session_id: str = None, payload=None
    ) -> List[str]:
        """Broadcast message to all peers, and return list of session id."""
        message = SessionMessage(
            tag=tag,
            source=self._name,
            destination="*",
            payload=payload,
            session_id=session_id,
            session_type=session_type
        )

        broadcast_status = self._driver.broadcast(message)

        if not broadcast_status:
            return [message.session_id] * len(self.get_peers())
        elif broadcast_status and self._is_enable_fault_tolerant:
            self._logger.warn(f"{self._name} failure to broadcast message to any peers, as {str(broadcast_status)}")
        else:
            raise broadcast_status

    def broadcast(
        self, tag: Union[str, Enum], session_type: SessionType, session_id: str = None, payload=None
    ) -> List[Message]:
        """Broadcast message to all peers, and return all replied messages.

        Args:
            tag (str|Enum): Message's tag.
            session_type (Enum): Message's session type.
            session_id (str): Message's session id. Defaults to None.
            payload (object): The true data. Defaults to None.

        Returns:
            List[Message]: List of replied messages.
        """
        return self.receive_by_id(self._broadcast(tag, session_type, session_id, payload))

    def ibroadcast(
        self, tag: Union[str, Enum], session_type: SessionType, session_id: str = None, payload=None
    ) -> List[str]:
        """Broadcast message to all subscribers, and return list of message's session id.

        Args:
            tag (str|Enum): Message's tag.
            session_type (Enum): Message's session type.
            session_id (str): Message's session id. Defaults to None.
            payload (object): The true data. Defaults to None.

        Returns:
            List[str]: List of message's session id which related to the replied message.
        """
        return self._broadcast(tag, session_type, session_id, payload)

    def isend(self, message: Message) -> List[str]:
        """Send a message to a remote peer.

        Args:
            message: Message to be sent.

        Returns:
            List[str]: List of message's session id.
        """
        sending_status = self._driver.send(message)

        if not sending_status:
            return [message.session_id]
        elif sending_status and self._is_enable_fault_tolerant:
            self._logger.warn(
                f"{self._name} failure to send message to {message.destination}, as {str(sending_status)}")
        else:
            raise sending_status

    def send(self, message: Message) -> List[Message]:
        """Send a message to a remote peer.

        Args:
            message: Message to be sent.

        Returns:
            List[Message]: List of replied message.
        """
        sending_status = self._driver.send(message)

        if not sending_status:
            return self.receive_by_id([message.session_id])
        elif sending_status and self._is_enable_fault_tolerant:
            self._logger.warn(
                f"{self._name} failure to send message to {message.destination}, as {str(sending_status)}"
            )
        else:
            raise sending_status

    def reply(
        self, received_message: SessionMessage, tag: Union[str, Enum] = None, payload=None, ack_reply: bool = False
    ) -> List[str]:
        """Reply a received message.

        Args:
            received_message (Message): The message need to reply.
            tag (str|Enum): New message tag, if None, keeps the original message's tag. Defaults to None.
            payload (object): New message payload, if None, keeps the original message's payload. Defaults to None.
            ack_reply (bool): If True, it is acknowledge reply. Defaults to False.

        Returns:
            List[str]: Message belonged session id.
        """
        if received_message.session_type == SessionType.TASK:
            session_stage = TaskSessionStage.RECEIVE if ack_reply else TaskSessionStage.COMPLETE
        else:
            session_stage = NotificationSessionStage.RECEIVE

        replied_message = SessionMessage(
            tag=tag if tag else received_message.tag,
            source=self._name,
            destination=received_message.source,
            session_id=received_message.session_id,
            payload=payload,
            session_stage=session_stage
        )
        return self.isend(replied_message)

    def forward(
        self, received_message: SessionMessage, destination: str, tag: Union[str, Enum] = None, payload=None
    ) -> List[str]:
        """Forward a received message.

        Args:
            received_message (Message): The message need to forward.
            destination (str): The receiver of message.
            tag (str|Enum): New message tag, if None, keeps the original message's tag. Defaults to None.
            payload (object): Message payload, if None, keeps the original message's payload. Defaults to None.

        Returns:
            List[str]: Message belonged session id.
        """
        forward_message = SessionMessage(
            tag=tag if tag else received_message.tag,
            source=self._name,
            destination=destination,
            session_id=received_message.session_id,
            payload=payload if payload else received_message.payload,
            session_stage=received_message.session_stage
        )
        return self.isend(forward_message)
