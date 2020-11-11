# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import subprocess
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed

from maro.communication import Proxy, SessionMessage, dist

from utils import get_random_port, proxy_generator


def handler_function(that, proxy, message):
    replied_payload = {"counter": message.payload["counter"] + 1}
    proxy.reply(message, payload=replied_payload)
    sys.exit(0)


def lunch_receiver(handler_dict, redis_port):
    proxy = proxy_generator("receiver", redis_port)

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
        # Initial Redis.
        redis_port = get_random_port()
        cls.redis_process = subprocess.Popen(["redis-server", "--port", str(redis_port), "--daemonize yes"])
        cls.redis_process.wait()

        # Initial receiver.
        conditional_event = "sender:*:1"
        handler_dict = {conditional_event: handler_function}
        decorator_task = threading.Thread(target=lunch_receiver, args=(handler_dict, redis_port, ))
        decorator_task.start()

        # Initial sender proxy.
        with ThreadPoolExecutor() as executor:
            sender_task = executor.submit(proxy_generator, "sender", redis_port)
            cls.sender_proxy = sender_task.result()

    @classmethod
    def tearDownClass(cls) -> None:
        print(f"The dist decorator unit test finished!")
        if hasattr(cls, "redis_process"):
            cls.redis_process.kill()

    def test_decorator(self):
        message = SessionMessage(tag="unittest",
                                 source=TestDecorator.sender_proxy.component_name,
                                 destination=TestDecorator.sender_proxy.peers["receiver"][0],
                                 payload={"counter": 0})
        replied_message = TestDecorator.sender_proxy.send(message)

        self.assertEqual(message.payload["counter"]+1, replied_message[0].payload["counter"])


if __name__ == "__main__":
    unittest.main()
