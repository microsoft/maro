# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed

from maro.communication import SessionMessage, ZmqDriver


def message_receive(driver):
    for received_message in driver.receive(is_continuous=False):
        return received_message.payload

@unittest.skipUnless(os.environ.get("test_with_zmq", False), "require zmq")
class TestDriver(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(f"The ZMQ driver unit test start!")
        cls.peer_list = ["receiver_1", "receiver_2", "receiver_3"]
        # Initial send driver.
        cls.sender = ZmqDriver()
        sender_address = cls.sender.address

        # Initial receive drivers.
        cls.receivers = {}
        receiver_addresses = {}
        for peer in cls.peer_list:
            peer_driver = ZmqDriver()
            peer_driver.connect({"sender": sender_address})
            cls.receivers[peer] = peer_driver
            receiver_addresses[peer] = peer_driver.address

        cls.sender.connect(receiver_addresses)

    @classmethod
    def tearDownClass(cls) -> None:
        print(f"The ZMQ driver unit test finished!")

    def test_send(self):
        for peer in TestDriver.peer_list:
            message = SessionMessage(tag="unit_test",
                                     source="sender",
                                     destination=peer,
                                     payload="hello_world")
            TestDriver.sender.send(message)

            for received_message in TestDriver.receivers[peer].receive(is_continuous=False):
                self.assertEqual(received_message.payload, message.payload)

    def test_broadcast(self):
        with ThreadPoolExecutor(max_workers=len(TestDriver.peer_list)) as executor:
            all_task = [executor.submit(message_receive, (TestDriver.receivers[peer])) for peer in TestDriver.peer_list]

            message = SessionMessage(tag="unit_test",
                                     source="sender",
                                     destination="*",
                                     payload="hello_world")
            TestDriver.sender.broadcast(message)

            for task in all_task:
                res = task.result()
                self.assertEqual(res, message.payload)


if __name__ == "__main__":
    unittest.main()
