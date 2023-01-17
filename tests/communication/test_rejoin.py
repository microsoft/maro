# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import multiprocessing
import os
import subprocess
import sys
import time
import unittest

from maro.communication import Proxy, SessionMessage, SessionType

from tests.communication.utils import get_random_port

PROXY_PARAMETER = {
    "group_name": "communication_unit_test",
    "enable_rejoin": True,
    "peers_catch_lifetime": 1,
    "minimal_peers": {"actor": 1, "master": 1},
    "enable_message_cache_for_rejoin": True,
    "timeout_for_minimal_peer_number": 300,
}


def actor_init(queue, redis_port):
    proxy = Proxy(
        component_type="actor", expected_peers={"master": 1}, redis_address=("localhost", redis_port), **PROXY_PARAMETER
    )

    # Continuously receive messages from proxy.
    for msg in proxy.receive():
        print(f"receive message from master. {msg.tag}")
        if msg.tag == "cont":
            proxy.reply(message=msg, tag="recv", body="successful receive!")
        elif msg.tag == "stop":
            proxy.reply(message=msg, tag="recv", body=f"{proxy.name} exited!")
            queue.put(proxy.name)
            break
        elif msg.tag == "finish":
            proxy.reply(message=msg, tag="recv", body=f"{proxy.name} finish!")
            sys.exit(0)

    proxy.close()
    sys.exit(1)


def fake_rejoin(queue, redis_port):
    component_name = queue.get()
    time.sleep(5)
    os.environ["COMPONENT_NAME"] = component_name
    actor_init(queue, redis_port)


@unittest.skipUnless(os.environ.get("test_with_redis", False), "require redis")
class TestRejoin(unittest.TestCase):
    master_proxy = None

    @classmethod
    def setUpClass(cls):
        print(f"The proxy unit test start!")
        # Initialize the Redis.
        redis_port = get_random_port()
        cls.redis_process = subprocess.Popen(["redis-server", "--port", str(redis_port), "--daemonize yes"])
        cls.redis_process.wait()

        # Initialize the proxies.
        cls.peers_number = 3
        q = multiprocessing.Queue()
        actor_process_list = []
        for i in range(cls.peers_number):
            actor_process_list.append(multiprocessing.Process(target=actor_init, args=(q, redis_port)))
        process_rejoin = multiprocessing.Process(target=fake_rejoin, args=(q, redis_port))

        for i in range(cls.peers_number):
            actor_process_list[i].start()
        process_rejoin.start()

        cls.master_proxy = Proxy(
            component_type="master",
            expected_peers={"actor": 3},
            redis_address=("localhost", redis_port),
            **PROXY_PARAMETER,
        )

        cls.peers = cls.master_proxy.peers["actor"]

    @classmethod
    def tearDownClass(cls) -> None:
        print(f"The proxy unit test finished!")
        cls.master_proxy.ibroadcast(component_type="actor", tag="finish", session_type=SessionType.NOTIFICATION)
        if hasattr(cls, "redis_process"):
            cls.redis_process.kill()

    def test_rejoin(self):
        # Check all connected.
        destination_payload_list = []
        for peer in TestRejoin.peers:
            destination_payload_list.append((peer, "continuous"))

        # Connection check.
        replied = TestRejoin.master_proxy.scatter(
            tag="cont",
            session_type=SessionType.NOTIFICATION,
            destination_payload_list=destination_payload_list,
        )
        self.assertEqual(len(replied), TestRejoin.peers_number)

        # Disconnect one peer.
        disconnect_message = SessionMessage(
            tag="stop",
            source=TestRejoin.master_proxy.name,
            destination=TestRejoin.peers[1],
            body=None,
            session_type=SessionType.TASK,
        )
        TestRejoin.master_proxy.isend(disconnect_message)

        # Now, 1 peer exited, only have 2 peers.
        time.sleep(2)
        replied = TestRejoin.master_proxy.scatter(
            tag="cont",
            session_type=SessionType.NOTIFICATION,
            destination_payload_list=destination_payload_list,
        )
        self.assertEqual(len(replied), TestRejoin.peers_number - 1)

        # Wait for rejoin.
        time.sleep(5)
        # Now, all peers rejoin.
        replied = TestRejoin.master_proxy.scatter(
            tag="cont",
            session_type=SessionType.NOTIFICATION,
            destination_payload_list=destination_payload_list,
        )
        self.assertEqual(len(replied), TestRejoin.peers_number + 1)


if __name__ == "__main__":
    unittest.main()
