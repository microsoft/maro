# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# native lib
import json
import os
import signal
import sys
import time
import uuid
from collections import defaultdict, deque, namedtuple
from enum import Enum
from typing import Dict, List, Tuple, Union

# third party lib
import redis

# private lib
from maro.utils import DummyLogger, InternalLogger
from maro.utils.exception.communication_exception import InformationUncompletedError, PeersMissError, PendingToSend
from maro.utils.exit_code import KILL_ALL_EXIT_CODE, NON_RESTART_EXIT_CODE

from .driver import DriverType, ZmqDriver
from .message import Message, NotificationSessionStage, SessionMessage, SessionType, TaskSessionStage
from .utils import default_parameters

_PEER_INFO = namedtuple("PEER_INFO", ["hash_table_name", "expected_number"])
HOST = default_parameters.proxy.redis.host
PORT = default_parameters.proxy.redis.port
MAX_RETRIES = default_parameters.proxy.redis.max_retries
BASE_RETRY_INTERVAL = default_parameters.proxy.redis.base_retry_interval
DELAY_FOR_SLOW_JOINER = default_parameters.proxy.delay_for_slow_joiner
ENABLE_REJOIN = default_parameters.proxy.peer_rejoin.enable  # Only enable at real k8s cluster or grass cluster
PEERS_CATCH_LIFETIME = default_parameters.proxy.peer_rejoin.peers_catch_lifetime
ENABLE_MESSAGE_CACHE_FOR_REJOIN = default_parameters.proxy.peer_rejoin.enable_message_cache
TIMEOUT_FOR_MINIMAL_PEER_NUMBER = default_parameters.proxy.peer_rejoin.timeout_for_minimal_peer_number
MINIMAL_PEERS = default_parameters.proxy.peer_rejoin.minimal_peers
IS_REMOVE_FAILED_CONTAINER = default_parameters.proxy.peer_rejoin.is_remove_failed_container
MAX_REJOIN_TIMES = default_parameters.proxy.peer_rejoin.max_rejoin_times
MAX_LENGTH_FOR_MESSAGE_CACHE = default_parameters.proxy.peer_rejoin.max_length_for_message_cache


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
        retry_interval_base_value (float): The time interval between attempts. Defaults to 0.1.
        log_enable (bool): Open internal logger or not. Defaults to True.
        enable_rejoin (bool): Allow peers rejoin or not. Defaults to False, and must use with maro cli.
        minimal_peers Union[int, dict]: The minimal number of peers for each peer type.
        peers_catch_lifetime (int): The lifetime for onboard peers' information.
        enable_message_cache_for_rejoin (bool): Enable message cache for failed peers or not. Default to False.
        max_length_for_message_cache (int): The maximal number of cached messages.
        timeout_for_minimal_peer_number (int): The timeout of waiting enough alive peers.
        is_remove_failed_container (bool): Enable remove failed containers automatically or not. Default to False.
        max_rejoin_times (int): The maximal retry times for one peer rejoins.
    """

    def __init__(
        self,
        group_name: str,
        component_type: str,
        expected_peers: dict,
        driver_type: DriverType = DriverType.ZMQ,
        driver_parameters: dict = None,
        redis_address: Tuple = (HOST, PORT),
        max_retries: int = MAX_RETRIES,
        retry_interval_base_value: float = BASE_RETRY_INTERVAL,
        log_enable: bool = True,
        enable_rejoin: bool = ENABLE_REJOIN,
        minimal_peers: Union[int, dict] = MINIMAL_PEERS,
        peers_catch_lifetime: int = PEERS_CATCH_LIFETIME,
        enable_message_cache_for_rejoin: bool = ENABLE_MESSAGE_CACHE_FOR_REJOIN,
        max_length_for_message_cache: int = MAX_LENGTH_FOR_MESSAGE_CACHE,
        timeout_for_minimal_peer_number: int = TIMEOUT_FOR_MINIMAL_PEER_NUMBER,
        is_remove_failed_container: bool = IS_REMOVE_FAILED_CONTAINER,
        max_rejoin_times: int = MAX_REJOIN_TIMES
    ):
        self._group_name = group_name
        self._component_type = component_type
        self._redis_hash_name = f"{self._group_name}:{self._component_type}"
        if "COMPONENT_NAME" in os.environ:
            self._name = os.getenv("COMPONENT_NAME")
        else:
            unique_id = str(uuid.uuid1()).replace("-", "")
            self._name = f"{self._component_type}_proxy_{unique_id}"
        self._max_retries = max_retries
        self._retry_interval_base_value = retry_interval_base_value
        self._log_enable = log_enable
        self._logger = InternalLogger(component_name=self._name) if self._log_enable else DummyLogger()

        # TODO:In multiprocess with spawn start method, the driver must be initiated before the Redis.
        # Otherwise it will cause Error 9: Bad File Descriptor in proxy.__del__(). Root cause not found.
        # Initialize the driver.
        if driver_type == DriverType.ZMQ:
            self._driver = ZmqDriver(
                component_type=self._component_type, **driver_parameters, logger=self._logger
            ) if driver_parameters else ZmqDriver(component_type=self._component_type, logger=self._logger)
        else:
            self._logger.error(f"Unsupported driver type {driver_type}, please use DriverType class.")
            sys.exit(NON_RESTART_EXIT_CODE)

        # Initialize the Redis.
        self._redis_connection = redis.Redis(host=redis_address[0], port=redis_address[1], socket_keepalive=True)
        try:
            self._redis_connection.ping()
        except Exception as e:
            self._logger.error(f"{self._name} failure to connect to redis server due to {e}")
            sys.exit(NON_RESTART_EXIT_CODE)

        # Record the peer's redis information.
        self._peers_info_dict = {}
        for peer_type, number in expected_peers.items():
            self._peers_info_dict[peer_type] = _PEER_INFO(
                hash_table_name=f"{self._group_name}:{peer_type}",
                expected_number=number
            )
        self._onboard_peer_dict = defaultdict(dict)

        # Temporary store the message.
        self._message_cache = defaultdict(list)

        # Parameters for dynamic peers.
        self._enable_rejoin = enable_rejoin
        self._is_remove_failed_container = is_remove_failed_container
        self._max_rejoin_times = max_rejoin_times
        if self._enable_rejoin:
            self._peers_catch_lifetime = peers_catch_lifetime
            self._timeout_for_minimal_peer_number = timeout_for_minimal_peer_number
            self._enable_message_cache = enable_message_cache_for_rejoin
            if self._enable_message_cache:
                self._message_cache_for_exited_peers = defaultdict(
                    lambda: deque([], maxlen=max_length_for_message_cache)
                )

            if isinstance(minimal_peers, int):
                self._minimal_peers = {
                    peer_type: max(minimal_peers, 1)
                    for peer_type, peer_info in self._peers_info_dict.items()
                }
            elif isinstance(minimal_peers, dict):
                self._minimal_peers = {
                    peer_type: max(minimal_peers[peer_type], 1)
                    for peer_type, peer_info in self._peers_info_dict.items()
                }
            else:
                self._logger.error("Unsupported minimal peers type, please use integer or dict.")
                sys.exit(NON_RESTART_EXIT_CODE)

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

        # Build component-container-mapping for dynamic component in k8s/grass cluster.
        if "JOB_ID" in os.environ and "CONTAINER_NAME" in os.environ:
            container_name = os.getenv("CONTAINER_NAME")
            job_id = os.getenv("JOB_ID")
            rejoin_config = {
                "rejoin:enable": int(self._enable_rejoin),
                "rejoin:max_restart_times": self._max_rejoin_times,
                "is_remove_failed_container": int(self._is_remove_failed_container)
            }

            self._redis_connection.hmset(f"job:{job_id}:runtime_details", rejoin_config)

            if self._enable_rejoin:
                self._redis_connection.hset(
                    f"job:{job_id}:rejoin_component_name_to_container_name", self._name, container_name
                )

    def __del__(self):
        self._redis_connection.hdel(self._redis_hash_name, self._name)
        self._driver.close()

    def _register_redis(self):
        """Self-registration on Redis and driver initialization.

        Redis store structure:
        Hash Table: name: group name + peer's type,
                    table (Dict[]): The key of table is the peer's name,
                                    the value of table is the peer's socket address.
        """
        self._redis_connection.hset(self._redis_hash_name, self._name, json.dumps(self._driver.address))

        # Handle interrupt signal for clearing Redis record.
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except Exception as e:
            self._logger.error(
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
                    self._logger.info(f"{self._name} successfully get all {peer_type}\'s name.")
                    break
                else:
                    self._logger.warn(
                        f"{self._name} failed to get {peer_type}\'s name. Retrying in "
                        f"{self._retry_interval_base_value * (2 ** retry_number)} seconds."
                    )
                    time.sleep(self._retry_interval_base_value * (2 ** retry_number))
                    retry_number += 1

            if not expected_peers_name:
                raise InformationUncompletedError(
                    f"{self._name} failure to get enough number of {peer_type} from redis."
                )

            self._onboard_peer_dict[peer_type] = {peer_name: None for peer_name in expected_peers_name}

        self._onboard_peers_start_time = time.time()

    def _build_connection(self):
        """Grabbing all peers' address from Redis, and connect all peers in driver."""
        for peer_type in self._peers_info_dict.keys():
            name_list = list(self._onboard_peer_dict[peer_type].keys())
            try:
                peers_socket_value = self._redis_connection.hmget(
                    self._peers_info_dict[peer_type].hash_table_name,
                    name_list
                )
                for idx, peer_name in enumerate(name_list):
                    self._onboard_peer_dict[peer_type][peer_name] = json.loads(peers_socket_value[idx])
                    self._logger.info(f"{self._name} successfully get {peer_name}\'s socket address")
            except Exception as e:
                raise InformationUncompletedError(f"{self._name} failed to get {name_list}\'s address. Due to {str(e)}")

            self._driver.connect(self._onboard_peer_dict[peer_type])

    @property
    def group_name(self) -> str:
        """str: Identifier for the group of all communication components."""
        return self._group_name

    @property
    def name(self) -> str:
        """str: Unique identifier in the current group."""
        return self._name

    @property
    def component_type(self) -> str:
        """str: Component's type in the current group."""
        return self._component_type

    @property
    def peers_name(self) -> Dict:
        """Dict: The ``Dict`` of all connected peers' names, stored by peer type."""
        return {
            peer_type: list(self._onboard_peer_dict[peer_type].keys()) for peer_type in self._peers_info_dict.keys()
        }

    def receive(self, is_continuous: bool = True, timeout: int = None):
        """Receive messages from communication driver.

        Args:
            is_continuous (bool): Continuously receive message or not. Defaults to True.
        """
        return self._driver.receive(is_continuous, timeout=timeout)

    def receive_by_id(self, targets: List[str], timeout: int = None) -> List[Message]:
        """Receive target messages from communication driver.

        Args:
            targets List[str]: List of ``session_id``.
                E.g. ['0_learner0_actor0', '1_learner1_actor1', ...].

        Returns:
            List[Message]: List of received messages.
        """
        if not isinstance(targets, list) and not isinstance(targets, str):
            # The input may be None, if enable peer rejoin.
            self._logger.warn(f"Unrecognized target {targets}.")
            return

        # Pre-process targets.
        if isinstance(targets, str):
            targets = [targets]
        pending_targets, received_messages = targets[:], []

        # Check message cache for saved messages.
        for session_id in targets:
            if session_id in self._message_cache:
                pending_targets.remove(session_id)
                received_messages.append(self._message_cache[session_id].pop(0))
                if not self._message_cache[session_id]:
                    del self._message_cache[session_id]

        if not pending_targets:
            return received_messages

        # Wait for incoming messages.
        for msg in self._driver.receive(is_continuous=True, timeout=timeout):
            if not msg:
                return received_messages

            if msg.session_id in pending_targets:
                pending_targets.remove(msg.session_id)
                received_messages.append(msg)
            else:
                self._message_cache[msg.session_id].append(msg)

            if not pending_targets:
                break

        return received_messages

    def _scatter(
        self,
        tag: Union[str, Enum],
        session_type: SessionType,
        destination_payload_list: list
    ) -> List[str]:
        """Scatters a list of data to peers, and return list of session id."""
        session_id_list = []

        for destination, payload in destination_payload_list:
            message = SessionMessage(
                tag=tag,
                source=self._name,
                destination=destination,
                payload=payload,
                session_type=session_type
            )
            send_result = self.isend(message)
            if isinstance(send_result, list):
                session_id_list += send_result

        return session_id_list

    def scatter(
        self,
        tag: Union[str, Enum],
        session_type: SessionType,
        destination_payload_list: list,
        timeout: int = -1
    ) -> List[Message]:
        """Scatters a list of data to peers, and return replied messages.

        Args:
            tag (str|Enum): Message's tag.
            session_type (Enum): Message's session type.
            destination_payload_list ([Tuple(str, object)]): The destination-payload list.
                The first item of the tuple in list is the message destination,
                and the second item of the tuple in list is the message payload.

        Returns:
            List[Message]: List of replied message.
        """
        return self.receive_by_id(
            targets=self._scatter(tag, session_type, destination_payload_list),
            timeout=timeout
        )

    def iscatter(
        self,
        tag: Union[str, Enum],
        session_type: SessionType,
        destination_payload_list: list
    ) -> List[str]:
        """Scatters a list of data to peers, and return list of message id.

        Args:
            tag (str|Enum): Message's tag.
            session_type (Enum): Message's session type.
            destination_payload_list ([Tuple(str, object)]): The destination-payload list.
                The first item of the tuple in list is the message's destination,
                and the second item of the tuple in list is the message's payload.

        Returns:
            List[str]: List of message's session id.
        """
        return self._scatter(tag, session_type, destination_payload_list)

    def _broadcast(
        self,
        component_type: str,
        tag: Union[str, Enum],
        session_type: SessionType,
        payload=None
    ) -> List[str]:
        """Broadcast message to all peers, and return list of session id."""
        if component_type not in list(self._onboard_peer_dict.keys()):
            self._logger.error(
                f"peer_type: {component_type} cannot be recognized. Please check the input of proxy.broadcast."
            )
            sys.exit(NON_RESTART_EXIT_CODE)

        if self._enable_rejoin:
            self._rejoin(component_type)

        message = SessionMessage(
            tag=tag,
            source=self._name,
            destination=component_type,
            payload=payload,
            session_type=session_type
        )

        self._driver.broadcast(component_type, message)

        return [message.session_id for _ in range(len(self._onboard_peer_dict[component_type]))]

    def broadcast(
        self,
        component_type: str,
        tag: Union[str, Enum],
        session_type: SessionType,
        payload=None,
        timeout: int = None
    ) -> List[Message]:
        """Broadcast message to all peers, and return all replied messages.

        Args:
            component_type (str): Broadcast to all peers in this type.
            tag (str|Enum): Message's tag.
            session_type (Enum): Message's session type.
            payload (object): The true data. Defaults to None.

        Returns:
            List[Message]: List of replied messages.
        """
        return self.receive_by_id(
            targets=self._broadcast(component_type, tag, session_type, payload),
            timeout=timeout
        )

    def ibroadcast(
        self,
        component_type: str,
        tag: Union[str, Enum],
        session_type: SessionType,
        payload=None
    ) -> List[str]:
        """Broadcast message to all subscribers, and return list of message's session id.

        Args:
            component_type (str): Broadcast to all peers in this type.
            tag (str|Enum): Message's tag.
            session_type (Enum): Message's session type.
            payload (object): The true data. Defaults to None.

        Returns:
            List[str]: List of message's session id which related to the replied message.
        """
        return self._broadcast(component_type, tag, session_type, payload)

    def _send(self, message: Message) -> Union[List[str], None]:
        """Send a message to a remote peer.

        Args:
            message: Message to be sent.

        Returns:
            Union[List[str], None]: The list of message's session id;
                If enable rejoin, it will return None when sending message to the failed peers;
                If enable rejoin and message cache, it may return list of session id which from
                the pending messages in message cache.
        """
        session_id_list = []
        if self._enable_rejoin:
            peer_type = self.get_peer_type(message.destination)
            self._rejoin(peer_type)

            # Check message cache.
            if (
                self._enable_message_cache
                and message.destination in list(self._onboard_peer_dict[peer_type].keys())
                and message.destination in list(self._message_cache_for_exited_peers.keys())
            ):
                self._logger.info(f"Sending pending message to {message.destination}.")
                for pending_message in self._message_cache_for_exited_peers[message.destination]:
                    self._driver.send(pending_message)
                    session_id_list.append(pending_message.session_id)
                del self._message_cache_for_exited_peers[message.destination]

        try:
            self._driver.send(message)
            session_id_list.append(message.session_id)
            return session_id_list
        except PendingToSend as e:
            self._logger.warn(f"{e} Peer {message.destination} exited, but still have enough peers.")
            if self._enable_message_cache:
                self._push_message_to_message_cache(message)

    def isend(self, message: Message) -> Union[List[str], None]:
        """Send a message to a remote peer.

        Args:
            message: Message to be sent.

        Returns:
            Union[List[str], None]: The list of message's session id;
                If enable rejoin, it will return None when sending message to the failed peers.
                If enable rejoin and message cache, it may return list of session id which from
                the pending messages.
        """
        return self._send(message)

    def send(
        self,
        message: Message,
        timeout: int = None
    ) -> Union[List[Message], None]:
        """Send a message to a remote peer.

        Args:
            message: Message to be sent.

        Returns:
            Union[List[Message], None]: The list of received message;
                If enable rejoin, it will return None when sending message to the failed peers.
                If enable rejoin and message cache, it may return list of messages which from
                the pending messages.
        """
        return self.receive_by_id(self._send(message), timeout)

    def reply(
        self,
        message: Union[SessionMessage, Message],
        tag: Union[str, Enum] = None,
        payload=None,
        ack_reply: bool = False
    ) -> List[str]:
        """Reply a received message.

        Args:
            message (Message): The message need to reply.
            tag (str|Enum): New message tag, if None, keeps the original message's tag. Defaults to None.
            payload (object): New message payload, if None, keeps the original message's payload. Defaults to None.
            ack_reply (bool): If True, it is acknowledge reply. Defaults to False.

        Returns:
            List[str]: Message belonged session id.
        """
        message.reply(tag=tag, payload=payload)
        if isinstance(message, SessionMessage):
            if message.session_type == SessionType.TASK:
                session_stage = TaskSessionStage.RECEIVE if ack_reply else TaskSessionStage.COMPLETE
            else:
                session_stage = NotificationSessionStage.RECEIVE
            message.session_stage = session_stage

        return self.isend(message)

    def forward(
        self,
        message: Union[SessionMessage, Message],
        destination: str,
        tag: Union[str, Enum] = None,
        payload=None
    ) -> List[str]:
        """Forward a received message.

        Args:
            message (Message): The message need to forward.
            destination (str): The receiver of message.
            tag (str|Enum): New message tag, if None, keeps the original message's tag. Defaults to None.
            payload (object): Message payload, if None, keeps the original message's payload. Defaults to None.

        Returns:
            List[str]: Message belonged session id.
        """
        message.forward(destination=destination, tag=tag, payload=payload)
        return self.isend(message)

    def _check_peers_update(self):
        """Compare the peers' information on local with the peers' information on Redis.

        For the different between two peers information dict's key (peer_name):
            If some peers only appear on Redis, the proxy will connect with those peers;
            If some peers only appear on local, the proxy will disconnect with those peers;
        For the different between two peers information dict's value (peers_socket_address):
            If some peers' information is different between Redis and local, the proxy will update those
            peers (driver disconnect with the local information and connect with the peer's information on Redis).
        """
        for peer_type in self._peers_info_dict.keys():
            # onboard_peers_dict_on_redis is the newest peers' information from the Redis.
            onboard_peers_dict_on_redis = self._redis_connection.hgetall(
                self._peers_info_dict[peer_type].hash_table_name
            )
            onboard_peers_dict_on_redis = {
                key.decode(): json.loads(value) for key, value in onboard_peers_dict_on_redis.items()
            }

            # Mappings (instances of dict) compare equal if and only if they have equal (key, value) pairs.
            # Equality comparison of the keys and values enforces reflexivity.
            if self._onboard_peer_dict[peer_type] != onboard_peers_dict_on_redis:
                union_peer_name = list(self._onboard_peer_dict[peer_type].keys()) + \
                    list(onboard_peers_dict_on_redis.keys())
                for peer_name in union_peer_name:
                    # Add new peers (new key added on redis).
                    if peer_name not in list(self._onboard_peer_dict[peer_type].keys()):
                        self._logger.info(f"PEER_REJOIN: New peer {peer_name} join.")
                        self._driver.connect({peer_name: onboard_peers_dict_on_redis[peer_name]})
                        self._onboard_peer_dict[peer_type][peer_name] = onboard_peers_dict_on_redis[peer_name]
                    # Delete out of date peers (old key deleted on local)
                    elif peer_name not in onboard_peers_dict_on_redis.keys():
                        self._logger.info(f"PEER_REJOIN: Peer {peer_name} exited.")
                        self._driver.disconnect({peer_name: self._onboard_peer_dict[peer_type][peer_name]})
                        del self._onboard_peer_dict[peer_type][peer_name]
                    else:
                        # Peer's ip/port updated, re-connect (value update on redis).
                        if onboard_peers_dict_on_redis[peer_name] != self._onboard_peer_dict[peer_type][peer_name]:
                            self._logger.info(f"PEER_REJOIN: Peer {peer_name} rejoin.")
                            self._driver.disconnect({peer_name: self._onboard_peer_dict[peer_type][peer_name]})
                            self._driver.connect({peer_name: onboard_peers_dict_on_redis[peer_name]})
                            self._onboard_peer_dict[peer_type][peer_name] = onboard_peers_dict_on_redis[peer_name]

    def _rejoin(self, peer_type=None):
        """The logic about proxy rejoin.

        Update onboard peers with the peers on Redis, if onboard peers expired.
        If there are not enough peers for the given peer type, block until peers rejoin or timeout.
        """
        current_time = time.time()

        if current_time - self._onboard_peers_start_time > self._peers_catch_lifetime:
            self._check_peers_update()
            self._onboard_peers_start_time = current_time

        if len(self._onboard_peer_dict[peer_type].keys()) < self._minimal_peers[peer_type]:
            self._wait_for_minimal_peer_number(peer_type)

    def _wait_for_minimal_peer_number(self, peer_type):
        """Blocking until there are enough peers for the given peer type."""
        start_time = time.time()

        while time.time() - start_time < self._timeout_for_minimal_peer_number:
            self._logger.warn(
                f"No enough peers in {peer_type}! Wait for some peer restart. Remaining time: "
                f"{start_time + self._timeout_for_minimal_peer_number - time.time()}"
            )
            self._check_peers_update()

            if len(self._onboard_peer_dict[peer_type]) >= self._minimal_peers[peer_type]:
                return

            time.sleep(self._peers_catch_lifetime)

        self._logger.error(f"Failure to get enough peers for {peer_type}. All components will exited.")
        sys.exit(KILL_ALL_EXIT_CODE)

    def get_peer_type(self, peer_name: str) -> str:
        """Get peer type from given peer name.

        Args:
            peer_name (str): The component name of a peer, which form by peer_type and UUID.

        Returns:
            str: The component type of a peer in current group.
        """
        # peer_name is consist by peer_type + '_proxy_' + UUID where UUID is only compose by digits and letters.
        # So the peer_type must be the part before last '_proxy_'.
        peer_type = peer_name[:peer_name.rfind("_proxy_")]

        if peer_type not in list(self._onboard_peer_dict.keys()):
            self._logger.error(
                f"The message's destination {peer_name} does not belong to any recognized peer type. "
                f"Please check the input of message."
            )
            sys.exit(NON_RESTART_EXIT_CODE)

        return peer_type

    def _push_message_to_message_cache(self, message: Message):
        peer_name = message.destination
        for pending_message in self._message_cache_for_exited_peers[peer_name]:
            if pending_message.message_id == message.message_id:
                self._logger.warn(f"{message.message_id} has been added in the message cache.")
                return

        self._message_cache_for_exited_peers[peer_name].append(message)
        self._logger.info(f"Temporarily save message {message.session_id} to message cache.")

    def close(self):
        self._redis_connection.hdel(self._redis_hash_name, self._name)
        self._driver.close()
