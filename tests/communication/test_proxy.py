# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import subprocess
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed

from maro.communication import Proxy, SessionMessage, SessionType

from utils import get_random_port, proxy_generator


def message_receive(proxy):
    for received_message in proxy.receive(is_continuous=False):
        return received_message.payload


@unittest.skipUnless(os.environ.get("test_with_redis", False), "require redis")
class TestProxy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print(f"The proxy unit test start!")
        # Initial Redis
        redis_port = get_random_port()
        cls.redis_process = subprocess.Popen(["redis-server", "--port", str(redis_port), "--daemonize yes"])
        cls.redis_process.wait()

        # Initial proxies
        cls.worker_proxies = []
        proxy_type_list = ["master"] + ["worker"] * 5
        with ThreadPoolExecutor(max_workers=6) as executor:
            all_tasks = [executor.submit(proxy_generator, proxy_type, redis_port) for proxy_type in proxy_type_list]

            for task in as_completed(all_tasks):
                result = task.result()
                if "master" in result.component_name:
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
            send_msg = SessionMessage(tag="unit_test",
                                      source=TestProxy.master_proxy.component_name,
                                      destination=worker_proxy.component_name,
                                      payload="hello_world!")
            TestProxy.master_proxy.isend(send_msg)

            for receive_message in worker_proxy.receive(is_continuous=False):
                self.assertEqual(send_msg.payload, receive_message.payload)

    def test_scatter(self):
        scatter_payload = ["worker_1", "worker_2", "worker_3", "worker_4", "worker_5"]
        destination_payload_list = [(worker_proxy.component_name, scatter_payload[i])
                                    for i, worker_proxy in enumerate(TestProxy.worker_proxies)]

        TestProxy.master_proxy.iscatter(tag="unit_test",
                                        session_type=SessionType.NOTIFICATION,
                                        destination_payload_list=destination_payload_list)

        for i, worker_proxy in enumerate(TestProxy.worker_proxies):
            for msg in worker_proxy.receive(is_continuous=False):
                self.assertEqual(scatter_payload[i], msg.payload)

    def test_broadcast(self):
        with ThreadPoolExecutor(max_workers=len(TestProxy.worker_proxies)) as executor:
            all_tasks = [executor.submit(message_receive, worker_proxy) for worker_proxy in TestProxy.worker_proxies]

            payload = ["broadcast_unit_test"]
            TestProxy.master_proxy.ibroadcast(tag="unit_test",
                                              session_type=SessionType.NOTIFICATION,
                                              payload=payload)

            for task in all_tasks:
                result = task.result()
                self.assertEqual(result, payload)

    def test_reply(self):
        for worker_proxy in TestProxy.worker_proxies:
            send_msg = SessionMessage(tag="unit_test",
                                      source=TestProxy.master_proxy.component_name,
                                      destination=worker_proxy.component_name,
                                      payload="hello ")
            session_id_list = TestProxy.master_proxy.isend(send_msg)

            for receive_message in worker_proxy.receive(is_continuous=False):
                worker_proxy.reply(received_message=receive_message, tag="unit_test", payload="world!")

            replied_msg_list = TestProxy.master_proxy.receive_by_id(session_id_list)
            self.assertEqual(send_msg.payload + replied_msg_list[0].payload, "hello world!")


if __name__ == "__main__":
    unittest.main()
