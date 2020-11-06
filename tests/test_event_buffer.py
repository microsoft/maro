# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import unittest

from maro.event_buffer import Event, EventBuffer, EventCategory, EventState


class TestEventBuffer(unittest.TestCase):
    def setUp(self):
        self.eb = EventBuffer()

    def test_gen_event(self):
        """Test event generating correct"""
        evt = self.eb.gen_atom_event(1, 1, (0, 0))

        # fields should be same as specified
        self.assertEqual(evt.category, EventCategory.ATOM)
        self.assertEqual(evt.tick, 1)
        self.assertEqual(evt.event_type , 1)
        self.assertEqual(evt.payload, (0, 0))

        evt = self.eb.gen_cascade_event(2, 2, (1, 1, 1))

        self.assertEqual(evt.category, EventCategory.CASCADE)
        self.assertEqual(evt.tick, 2)
        self.assertEqual(evt.event_type , 2)
        self.assertEqual(evt.payload, (1, 1, 1))

    def test_insert_event(self):
        """Test insert event works as expected"""

        # pending pool should be empty at beginning
        self.assertEqual(len(self.eb._pending_events), 0)

        evt = self.eb.gen_atom_event(1, 1, 1)

        self.eb.insert_event(evt)

        # after insert one event, we should have 1 in pending pool
        self.assertEqual(len(self.eb._pending_events), 1)

    def test_event_dispatch(self):
        """Test event dispatching work as expected"""
        def cb(evt):
            # test event tick
            self.assertEqual(1, evt.tick, msg="recieved event tick should be 1")

            # test event payload
            self.assertTupleEqual((1, 3), evt.payload, msg="recieved event's payload should be (1, 3)")

        evt = self.eb.gen_atom_event(1, 1, (1, 3))

        self.eb.insert_event(evt)

        self.eb.register_event_handler(1, cb)

        self.eb.execute(1) # dispatch event

    def test_get_finish_events(self):
        """Test if we can get correct finished events"""

        # no finised at first
        self.assertListEqual([], self.eb.get_finished_events(), msg="finished pool should be empty")

        evt = self.eb.gen_atom_event(1, 1, (1, 3))

        self.eb.insert_event(evt)

        self.eb.execute(1)

        # after dispatching, finish pool should contains 1 object
        self.assertEqual(1, len(self.eb.get_finished_events()), msg="after dispathing, there should 1 object")

    def test_get_pending_events(self):
        """Test if we can get correct pending events"""

        # not pending at first
        self.assertEqual(0, len(self.eb.get_pending_events(1)), msg="pending pool should be empty")

        evt = self.eb.gen_atom_event(1, 1, (1, 3))

        self.eb.insert_event(evt)

        self.assertEqual(1, len(self.eb.get_pending_events(1)), msg="pending pool should contains 1 objects")

    def test_reset(self):
        """Test reset, all internal states should be reset"""
        evt = self.eb.gen_atom_event(1, 1, 1)

        self.eb.insert_event(evt)

        self.eb.reset()
        
        self.assertEqual(len(self.eb._pending_events), 0)
        self.assertEqual(len(self.eb._finished_events), 0)


if __name__ == "__main__":
    unittest.main()
