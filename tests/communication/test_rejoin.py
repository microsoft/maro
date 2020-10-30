# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import multiprocessing
import subprocess
import unittest
import os
import sys
import time

from maro.communication import Proxy, SessionMessage, SessionType
from tests.communication.utils import get_random_port

PROXY_PARAMETER = {
    "group_name": "communication_unit_test",
    "enable_rejoin": True,
    "peer_update_frequency": 1,
    "minimal_peers": {"actor": 1, "master": 1},
    "enable_message_cache_for_rejoin": False,
    "timeout_for_minimal_peer_number": 300
}


def actor_init(queue, redis_port):
    proxy = Proxy(
        component_type="actor",
        expected_peers={"master": 1},
        redis_address=("localhost", redis_port),
        **PROXY_PARAMETER
    )

    # continuously receive messages from proxy
    for msg in proxy.receive(is_continuous=True):
        print(f"receive message from master. {msg.tag}")
        if msg.tag == "cont":
            proxy.reply(received_message=msg, tag="recv", payload="successful receive!")
        elif msg.tag == "stop":
            proxy.reply(received_message=msg, tag="recv", payload=f"{proxy.component_name} exited!")
            queue.put(proxy.component_name)
            break
        elif msg.tag == "finish":
            proxy.reply(received_message=msg, tag="recv", payload=f"{proxy.component_name} finish!")
            sys.exit(0)

    proxy.__del__()
    sys.exit(1)


def fake_rejoin(queue, redis_port):
    component_name = queue.get()
    print(component_name)
    time.sleep(5)
    os.environ["COMPONENT_NAME"] = component_name
    actor_init(queue, redis_port)


@unittest.skipUnless(os.environ.get("test_with_redis", False), "require redis")
class TestRejoin(unittest.TestCase):
    master_proxy = None

    @classmethod
    def setUpClass(cls):
        print(f"The proxy unit test start!")
        # Initial Redis
        redis_port = get_random_port()
        cls.redis_process = subprocess.Popen(["redis-server", "--port", str(redis_port), "--daemonize yes"])
        cls.redis_process.wait()

        # Initial proxies
        q = multiprocessing.Queue()
        process_act1 = multiprocessing.Process(target=actor_init, args=(q, redis_port))
        process_act2 = multiprocessing.Process(target=actor_init, args=(q, redis_port))
        process_act3 = multiprocessing.Process(target=actor_init, args=(q, redis_port))
        process_rejoin = multiprocessing.Process(target=fake_rejoin, args=(q, redis_port))
        process_act1.start()
        process_act2.start()
        process_act3.start()
        process_rejoin.start()

        cls.master_proxy = Proxy(
            component_type="master",
            expected_peers={"actor": 3},
            redis_address=("localhost", redis_port),
            **PROXY_PARAMETER
        )

        cls.peers = cls.master_proxy.peers["actor"]

    @classmethod
    def tearDownClass(cls) -> None:
        print(f"The proxy unit test finished!")
        cls.master_proxy.ibroadcast(tag="finish", session_type=SessionType.NOTIFICATION)
        if hasattr(cls, "redis_process"):
            cls.redis_process.kill()

    def test_rejoin(self):
        # Check all connected
        dp_list = []
        for peer in TestRejoin.peers:
            dp_list.append((peer, "continuous"))

        TestRejoin.master_proxy.scatter(
            tag="cont",
            session_type=SessionType.NOTIFICATION,
            destination_payload_list=dp_list
        )

        # Disconnect one peer
        dis_message = SessionMessage(
            tag="stop",
            source=TestRejoin.master_proxy.component_name,
            destination=TestRejoin.peers[1],
            payload=None,
            session_type=SessionType.TASK
        )
        TestRejoin.master_proxy.isend(dis_message)

        # Now, 1 peer exited, only have 2 peers
        time.sleep(2)
        replied = TestRejoin.master_proxy.scatter(
            tag="cont", session_type=SessionType.NOTIFICATION,
            destination_payload_list=dp_list
        )
        self.assertEqual(len(replied), 2)

        # Wait for rejoin
        time.sleep(5)
        # Now, all peers rejoin
        replied = TestRejoin.master_proxy.scatter(
            tag="cont", session_type=SessionType.NOTIFICATION,
            destination_payload_list=dp_list
        )
        self.assertEqual(len(replied), 3)


if __name__ == "__main__":
    unittest.main()
