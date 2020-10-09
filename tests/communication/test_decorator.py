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


def proxy_generator(component_type):
    if component_type == "receiver":
        proxy = Proxy(group_name="proxy_unit_test",
                      component_type="receiver",
                      expected_peers={"sender": 1},
                      log_enable=False)
    elif component_type == "sender":
        proxy = Proxy(group_name="proxy_unit_test",
                      component_type="sender",
                      expected_peers={"receiver": 1},
                      log_enable=False)
    return proxy

def handler_function(that, proxy, message):
    replied_payload = message.payload + 1
    proxy.reply(message, payload=replied_payload)
    sys.exit(0)

def decorator_start(proxy, handler_dict):
    @dist(proxy, handler_dict)
    class Receiver:
        def __init__(self):
            pass
    
    decorator = Receiver()
    decorator.launch()

@unittest.skipUnless(os.environ.get("test_with_redis", False), "require redis")
class TestDecorator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"The dist decorator unit test start!")
        # Prepare Redis.
        try:
            temp_redis = redis.Redis(host="localhost", port=6379)
            temp_redis.ping()
        except Exception as _:
            cls.redis_process = subprocess.Popen(['redis-server'])
            time.sleep(1)

        # Prepare proxy.
        proxy_type_list = ["receiver", "sender"]
        with ThreadPoolExecutor(max_workers=3) as executor:
            all_task = [executor.submit(proxy_generator, (proxy_type)) for proxy_type in proxy_type_list]

            for task in as_completed(all_task):
                r = task.result()
                if "receiver" in r.component_name:
                    cls.receiver = r
                else:
                    cls.sender = r
        
        # Start decorator
        conditional_event = "sender:*:1"
        handler_dict = {conditional_event:handler_function}
        decorator_task = threading.Thread(target=decorator_start, args=(cls.receiver, handler_dict,))
        decorator_task.start()

    @classmethod
    def tearDownClass(cls) -> None:
        print(f"The dist decorator unit test finished!")
        if hasattr(cls, "redis_process"):
            cls.redis_process.terminate()

    def test_decorator(self):
        message = SessionMessage(tag="unittest",
                                 source=self.sender.component_name,
                                 destination=self.receiver.component_name,
                                 payload=0)
        replied_message = self.sender.send(message)

        self.assertEqual(1, replied_message[0].payload)


if __name__ == "__main__":
    unittest.main()
