# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import unittest
from maro.simulator.event_buffer import EventBuffer, Event, EventState, EventTag

class TestGraph(unittest.TestCase):
    def setUp(self):
        self.eb = EventBuffer()

    def test_gen_event(self):
        evt = self.eb.gen_atom_event(1, 1, (0, 0))

        self.assertEqual(evt.tag, EventTag.ATOM)
        self.assertEqual(evt.tick, 1)
        self.assertEqual(evt.event_type , 1)
        self.assertEqual(evt.payload, (0, 0))

        evt = self.eb.gen_cascade_event(2, 2, (1, 1, 1))

        self.assertEqual(evt.tag, EventTag.CASCADE)
        self.assertEqual(evt.tick, 2)
        self.assertEqual(evt.event_type , 2)
        self.assertEqual(evt.payload, (1, 1, 1))

    def test_insert_event(self):
        self.assertEqual(len(self.eb._pending_events), 0)

        evt = self.eb.gen_atom_event(1, 1, 1)

        self.eb.insert_event(evt)

        # we are accessing internal member to check result
        self.assertEqual(len(self.eb._pending_events), 1)


    def test_reset(self):
        evt = self.eb.gen_atom_event(1, 1, 1)

        self.eb.insert_event(evt)

        self.eb.reset()
        
        self.assertEqual(len(self.eb._pending_events), 0)
        self.assertEqual(len(self.eb._finished_events), 0)


if __name__ == "__main__":
    unittest.main()