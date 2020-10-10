# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from concurrent.futures import ThreadPoolExecutor, as_completed
import redis
import subprocess
import time
import unittest
import os

from maro.communication import Proxy, SessionMessage, SessionType
from utils import get_random_port


def proxy_generator(component_type, redis_port):
    if component_type == "master":
        proxy = Proxy(group_name="proxy_unit_test",
                      component_type="master",
                      expected_peers={"worker": 5},
                      redis_address=("localhost", redis_port),
                      log_enable=False)
    elif component_type == "worker":
        proxy = Proxy(group_name="proxy_unit_test",
                      component_type="worker",
                      expected_peers={"master": 1},
                      redis_address=("localhost", redis_port),
                      log_enable=False)
    return proxy


def message_receive(proxy):
    for received_message in proxy.receive(is_continuous=False):
        return received_message.payload


@unittest.skipUnless(os.environ.get("test_with_redis", False), "require redis")
class TestProxy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print(f"The proxy unit test start!")
        # prepare Redis
        random_port = get_random_port()
        cls.redis_process = subprocess.Popen(["redis-server", "--port", str(random_port), "--daemonize yes"])
        cls.redis_process.wait()

        # prepare proxies
        cls.worker_proxy = []
        proxy_type_list = ["master"] + ["worker"] * 5
        with ThreadPoolExecutor(max_workers=6) as executor:
            all_task = [executor.submit(proxy_generator, proxy_type, random_port) for proxy_type in proxy_type_list]

            for task in as_completed(all_task):
                result = task.result()
                if "master" in result.component_name:
                    cls.master_proxy = result
                else:
                    cls.worker_proxy.append(result)

    @classmethod
    def tearDownClass(cls) -> None:
        print(f"The proxy unit test finished!")
        if hasattr(cls, "redis_process"):
            cls.redis_process.kill()

    def test_send(self):
        for worker in self.worker_proxy:
            send_msg = SessionMessage(tag="unit_test",
                                      source=self.master_proxy.component_name,
                                      destination=worker.component_name,
                                      payload="hello_world!")
            self.master_proxy.isend(send_msg)

            for receive_message in worker.receive(is_continuous=False):
                self.assertEqual(send_msg.payload, receive_message.payload)

    def test_scatter(self):
        scatter_payload = ["worker_1", "worker_2", "worker_3", "worker_4", "worker_5"]
        destination_payload_list = [(worker.component_name, scatter_payload[i])
                                    for i, worker in enumerate(self.worker_proxy)]

        self.master_proxy.iscatter(tag="unit_test",
                                   session_type=SessionType.NOTIFICATION,
                                   destination_payload_list=destination_payload_list)

        for i, worker in enumerate(self.worker_proxy):
            for msg in worker.receive(is_continuous=False):
                self.assertEqual(scatter_payload[i], msg.payload)

    def test_broadcast(self):
        executor = ThreadPoolExecutor(max_workers=len(self.worker_proxy))
        all_task = [executor.submit(message_receive, (worker)) for worker in self.worker_proxy]

        payload = ["broadcast_unit_test"]
        self.master_proxy.ibroadcast(tag="unit_test",
                                     session_type=SessionType.NOTIFICATION,
                                     payload=payload)

        for task in as_completed(all_task):
            res = task.result()
            self.assertEqual(res, payload)

    def test_reply(self):
        for worker in self.worker_proxy:
            send_msg = SessionMessage(tag="unit_test",
                                      source=self.master_proxy.component_name,
                                      destination=worker.component_name,
                                      payload="hello ")
            session_id_list = self.master_proxy.isend(send_msg)

            for receive_message in worker.receive(is_continuous=False):
                worker.reply(received_message=receive_message, tag="unit_test", payload="world!")

            replied_msg_list = self.master_proxy.receive_by_id(session_id_list)
            self.assertEqual(send_msg.payload + replied_msg_list[0].payload, "hello world!")


if __name__ == "__main__":
    unittest.main()
