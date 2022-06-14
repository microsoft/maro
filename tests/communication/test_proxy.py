# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import subprocess
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed

from maro.communication import SessionMessage, SessionType

from tests.communication.utils import get_random_port, proxy_generator


def message_receive(proxy):
    return proxy.receive_once().body


@unittest.skipUnless(os.environ.get("test_with_redis", False), "require redis")
class TestProxy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"The proxy unit test start!")
        # Initialize the Redis
        redis_port = get_random_port()
        cls.redis_process = subprocess.Popen(["redis-server", "--port", str(redis_port), "--daemonize yes"])
        cls.redis_process.wait()

        # Initialize the proxies
        cls.worker_proxies = []
        proxy_type_list = ["master"] + ["worker"] * 5
        with ThreadPoolExecutor(max_workers=6) as executor:
            all_tasks = [executor.submit(proxy_generator, proxy_type, redis_port) for proxy_type in proxy_type_list]

            for task in as_completed(all_tasks):
                result = task.result()
                if "master" in result.name:
                    cls.master_proxy = result
                else:
                    cls.worker_proxies.append(result)

    @classmethod
    def tearDownClass(cls) -> None:
        print(f"The proxy unit test finished!")
        if hasattr(cls, "redis_process"):
            cls.redis_process.kill()

    def test_send(self):
        for worker_proxy in TestProxy.worker_proxies:
            send_msg = SessionMessage(
                tag="unit_test",
                source=TestProxy.master_proxy.name,
                destination=worker_proxy.name,
                body="hello_world!",
            )
            TestProxy.master_proxy.isend(send_msg)

            recv_msg = worker_proxy.receive_once()
            self.assertEqual(send_msg.body, recv_msg.body)

    def test_scatter(self):
        scatter_payload = ["worker_1", "worker_2", "worker_3", "worker_4", "worker_5"]
        destination_payload_list = [
            (worker_proxy.name, scatter_payload[i]) for i, worker_proxy in enumerate(TestProxy.worker_proxies)
        ]

        TestProxy.master_proxy.iscatter(
            tag="unit_test",
            session_type=SessionType.NOTIFICATION,
            destination_payload_list=destination_payload_list,
        )

        for i, worker_proxy in enumerate(TestProxy.worker_proxies):
            msg = worker_proxy.receive_once()
            self.assertEqual(scatter_payload[i], msg.body)

    def test_broadcast(self):
        with ThreadPoolExecutor(max_workers=len(TestProxy.worker_proxies)) as executor:
            all_tasks = [executor.submit(message_receive, worker_proxy) for worker_proxy in TestProxy.worker_proxies]

            payload = ["broadcast_unit_test"]
            TestProxy.master_proxy.ibroadcast(
                component_type="worker",
                tag="unit_test",
                session_type=SessionType.NOTIFICATION,
                body=payload,
            )

            for task in all_tasks:
                result = task.result()
                self.assertEqual(result, payload)

    def test_reply(self):
        for worker_proxy in TestProxy.worker_proxies:
            send_msg = SessionMessage(
                tag="unit_test",
                source=TestProxy.master_proxy.name,
                destination=worker_proxy.name,
                body="hello ",
            )
            session_id_list = TestProxy.master_proxy.isend(send_msg)

            recv_message = worker_proxy.receive_once()
            worker_proxy.reply(message=recv_message, tag="unit_test", body="world!")

            replied_msg_list = TestProxy.master_proxy.receive_by_id(session_id_list)
            self.assertEqual(send_msg.body + replied_msg_list[0].body, "hello world!")


if __name__ == "__main__":
    unittest.main()
