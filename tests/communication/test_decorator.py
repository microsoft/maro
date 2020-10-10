# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import redis
import subprocess
import sys
import time
import threading
import unittest

from maro.communication import Proxy, SessionMessage, dist
from utils import get_random_port


def proxy_generator(component_type, redis_port):
    if component_type == "receiver":
        proxy = Proxy(group_name="proxy_unit_test",
                      component_type="receiver",
                      expected_peers={"sender": 1},
                      redis_address=("localhost", redis_port),
                      log_enable=False)
    elif component_type == "sender":
        proxy = Proxy(group_name="proxy_unit_test",
                      component_type="sender",
                      expected_peers={"receiver": 1},
                      redis_address=("localhost", redis_port),
                      log_enable=False)
    return proxy


def handler_function(that, proxy, message):
    replied_payload = {"counter": message.payload["counter"] + 1}
    proxy.reply(message, payload=replied_payload)
    sys.exit(0)


def lunch_receiver(proxy, handler_dict):
    @dist(proxy, handler_dict)
    class Receiver:
        def __init__(self):
            pass

    receiver = Receiver()
    receiver.launch()


@unittest.skipUnless(os.environ.get("test_with_redis", False), "require redis")
class TestDecorator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"The dist decorator unit test start!")
        # Prepare Redis.
        random_port = get_random_port()
        cls.redis_process = subprocess.Popen(["redis-server", "--port", str(random_port), "--daemonize yes"])
        cls.redis_process.wait()

        # Prepare proxy.
        proxy_type_list = ["receiver", "sender"]
        with ThreadPoolExecutor(max_workers=2) as executor:
            all_task = [executor.submit(proxy_generator, proxy_type, random_port) for proxy_type in proxy_type_list]

            for task in as_completed(all_task):
                result = task.result()
                if "receiver" in result.component_name:
                    cls.receiver_proxy = result
                else:
                    cls.sender_proxy = result

        # Start decorator
        conditional_event = "sender:*:1"
        handler_dict = {conditional_event: handler_function}
        decorator_task = threading.Thread(target=lunch_receiver, args=(cls.receiver_proxy, handler_dict,))
        decorator_task.start()

    @classmethod
    def tearDownClass(cls) -> None:
        print(f"The dist decorator unit test finished!")
        if hasattr(cls, "redis_process"):
            cls.redis_process.kill()

    def test_decorator(self):
        message = SessionMessage(tag="unittest",
                                 source=self.sender_proxy.component_name,
                                 destination=self.receiver_proxy.component_name,
                                 payload={"counter": 0})
        replied_message = self.sender_proxy.send(message)

        self.assertEqual(1, replied_message[0].payload["counter"])


if __name__ == "__main__":
    unittest.main()
