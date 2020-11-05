# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest
from collections import defaultdict

from maro.communication import RegisterTable, SessionMessage


def get_peers(peer_type: str = None):
    env = ["worker_a.1", "worker_a.2", "worker_a.3", "worker_a.4", "worker_a.5",
           "worker_b.1", "worker_b.2", "worker_b.3", "worker_b.4", "worker_b.5"]
    if not peer_type or peer_type == "*":
        return env

    target_peer = []
    for peer in env:
        if peer_type in peer:
            target_peer.append(peer_type)
    return target_peer


def handle_function():
    pass


class TestRegisterTable(unittest.TestCase):

    def setUp(self) -> None:
        print(f"clear register table before each test.")
        self.register_table = RegisterTable(get_peers)

    @classmethod
    def setUpClass(cls) -> None:
        print(f"The register table unit test start!")
        # Prepare message dict for test
        cls.message_dict = {"worker_a": defaultdict(list),
                            "worker_b": defaultdict(list)}

        worker_a_list = ["worker_a.1", "worker_a.2", "worker_a.3", "worker_a.4", "worker_a.5"]
        worker_b_list = ["worker_b.1", "worker_b.2", "worker_b.3", "worker_b.4", "worker_b.5"]
        tag_type = ["tag_a", "tag_b"]

        for source in worker_a_list:
            for tag in tag_type:
                message = SessionMessage(tag=tag,
                                         source=source,
                                         destination="test")
                cls.message_dict["worker_a"][tag].append(message)

        for source in worker_b_list:
            for tag in tag_type:
                message = SessionMessage(tag=tag,
                                         source=source,
                                         destination="test")
                cls.message_dict["worker_b"][tag].append(message)

    @classmethod
    def tearDownClass(cls) -> None:
        print(f"The register table unit test finished!")

    def test_unit_conditional_event(self):
        # Accept a message from worker_a with tag_a.
        unit_event_1 = "worker_a:tag_a:1"
        self.register_table.register_event_handler(unit_event_1, handle_function)
        for msg in TestRegisterTable.message_dict["worker_a"]["tag_a"]:
            # The message from worker_a with tag_a, it will trigger handler function each time.
            self.register_table.push(msg)
            self.assertIsNotNone(self.register_table.get())

        for msg in TestRegisterTable.message_dict["worker_b"]["tag_b"]:
            # The message from worker_b with tag_b, the register table won't be trigger anytime.
            self.register_table.push(msg)
            self.assertEqual(self.register_table.get(), [])

    def test_special_symbol(self):
        # Accept a message from worker_a with any tags.
        unit_event_2 = "worker_a:*:1"
        self.register_table.register_event_handler(unit_event_2, handle_function)
        for msg in TestRegisterTable.message_dict["worker_a"]["tag_a"] + TestRegisterTable.message_dict["worker_a"]["tag_b"]:
            # The message from worker_a with any tags, it will trigger handler function each time.
            self.register_table.push(msg)
            self.assertIsNotNone(self.register_table.get())

        for msg in TestRegisterTable.message_dict["worker_b"]["tag_a"]:
            # The message from worker_b with tag_a, it won't trigger handler function.
            self.register_table.push(msg)
            self.assertEqual(self.register_table.get(), [])

    def test_percentage_case(self):
        # Accept messages from any source with tag_a until the number of message reach 60% of source number.
        unit_event_2 = "*:tag_a:50%"
        self.register_table.register_event_handler(unit_event_2, handle_function)
        for idx, msg in enumerate(TestRegisterTable.message_dict["worker_a"]["tag_a"] +
                                  TestRegisterTable.message_dict["worker_b"]["tag_a"]):
            # The message with tag_a, it will trigger handler function until receiving 5 times.
            self.register_table.push(msg)
            if (idx + 1) % 5 == 0:
                self.assertIsNotNone(self.register_table.get())
            else:
                self.assertEqual(self.register_table.get(), [])

    def test_conditional_event(self):
        # Accept the combination of two messages: one from worker_a with tag_a, and one from worker_b with tag_a.
        and_conditional_event = ("worker_a:tag_a:1", "worker_b:tag_a:1", "AND")
        self.register_table.register_event_handler(and_conditional_event, handle_function)
        for idx, msg in enumerate(TestRegisterTable.message_dict["worker_a"]["tag_a"] +
                                  TestRegisterTable.message_dict["worker_b"]["tag_a"]):
            # The messages with tag_a from worker_a and worker_b, it will trigger handler function until the
            # combination be satisfied.
            self.register_table.push(msg)
            if idx >= 5:
                self.assertIsNotNone(self.register_table.get())
            else:
                self.assertEqual(self.register_table.get(), [])

        # Accept the message from worker_a with tag_a, or from worker_b with tag_a.
        or_conditional_event = ("worker_a:tag_a:1", "worker_b:tag_a:1", "OR")
        self.register_table.register_event_handler(or_conditional_event, handle_function)
        for idx, msg in enumerate(TestRegisterTable.message_dict["worker_a"]["tag_a"] +
                                  TestRegisterTable.message_dict["worker_b"]["tag_a"]):
            # The messages with tag_a from worker_a and worker_b, it will trigger handler function each time.
            self.register_table.push(msg)
            self.assertIsNotNone(self.register_table.get())

    def test_complicated_conditional_event(self):
        # Accept the combination of three messages: one from worker_a with tag_a, one from worker_b with tag_a,
        # and one from worker_a with tag_b.
        recurrent_conditional_event = (("worker_a:tag_a:1", "worker_b:tag_a:1", "AND"), "worker_a:tag_b:1", "AND")
        self.register_table.register_event_handler(recurrent_conditional_event, handle_function)

        for msg in TestRegisterTable.message_dict["worker_a"]["tag_a"] + \
                   TestRegisterTable.message_dict["worker_b"]["tag_a"]:
            self.register_table.push(msg)

        for msg in TestRegisterTable.message_dict["worker_a"]["tag_b"]:
            self.register_table.push(msg)
            self.assertIsNotNone(self.register_table.get())

    def test_multiple_trigger(self):
        # Accept a message from worker_a with tag_a
        unit_event_1 = "worker_a:tag_a:1"
        # Accept a message from worker_a with any tag.
        unit_event_2 = "worker_a:*:1"
        self.register_table.register_event_handler(unit_event_1, handle_function)
        self.register_table.register_event_handler(unit_event_2, handle_function)
        for msg in TestRegisterTable.message_dict["worker_a"]["tag_a"]:
            # For each message from worker_a with tag_a, it will trigger two handler functions, and both of them will
            # have the same message.
            self.register_table.push(msg)
            res = self.register_table.get()
            self.assertEqual(len(res), 2)
            self.assertEqual(res[0][1], res[1][1])


if __name__ == '__main__':
    unittest.main()
