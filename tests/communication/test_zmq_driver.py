# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed

from maro.communication import SessionMessage, ZmqDriver


def message_receive(driver):
    return driver.receive_once().body


@unittest.skipUnless(os.environ.get("test_with_zmq", False), "require zmq")
class TestDriver(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        print(f"The ZMQ driver unit test start!")
        cls.peer_list = ["receiver_1", "receiver_2", "receiver_3"]
        # Initialize send driver.
        cls.sender = ZmqDriver(component_type="sender")
        sender_address = cls.sender.address

        # Initialize receive drivers.
        cls.receivers = {}
        receiver_addresses = {}
        for peer in cls.peer_list:
            peer_driver = ZmqDriver(component_type="receiver")
            peer_driver.connect({"sender": sender_address})
            cls.receivers[peer] = peer_driver
            receiver_addresses[peer] = peer_driver.address

        cls.sender.connect(receiver_addresses)

    @classmethod
    def tearDownClass(cls) -> None:
        print(f"The ZMQ driver unit test finished!")

    def test_send(self):
        for peer in TestDriver.peer_list:
            message = SessionMessage(
                tag="unit_test",
                source="sender",
                destination=peer,
                body="hello_world",
            )
            TestDriver.sender.send(message)

            recv_message = TestDriver.receivers[peer].receive_once()
            self.assertEqual(recv_message.body, message.body)

    def test_broadcast(self):
        executor = ThreadPoolExecutor(max_workers=len(TestDriver.peer_list))
        all_task = [executor.submit(message_receive, (TestDriver.receivers[peer])) for peer in TestDriver.peer_list]

        message = SessionMessage(
            tag="unit_test",
            source="sender",
            destination="*",
            body="hello_world",
        )
        TestDriver.sender.broadcast(topic="receiver", message=message)

        for task in as_completed(all_task):
            res = task.result()
            self.assertEqual(res, message.body)


if __name__ == "__main__":
    unittest.main()
