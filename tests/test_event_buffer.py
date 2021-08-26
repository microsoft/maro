# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import unittest
from typing import Optional

from maro.event_buffer import ActualEvent, AtomEvent, CascadeEvent, DummyEvent, EventBuffer, EventState, MaroEvents
from maro.event_buffer.event_linked_list import EventLinkedList


class TestEventBuffer(unittest.TestCase):
    def setUp(self):
        self.eb = EventBuffer()

    def test_cascade_event(self):
        evt = CascadeEvent(None, None, None, None)
        self.assertEqual(type(evt.immediate_event_head), DummyEvent)
        self.assertIsNone(evt.immediate_event_head.next_event)
        self.assertIsNone(evt.immediate_event_tail)
        self.assertEqual(evt.immediate_event_head.next_event, evt.immediate_event_tail)
        self.assertEqual(evt.immediate_event_count, 0)

        evt.add_immediate_event(AtomEvent(1, None, None, None), is_head=False)
        evt.add_immediate_event(AtomEvent(2, None, None, None), is_head=False)
        evt.add_immediate_event(AtomEvent(3, None, None, None), is_head=True)
        evt.add_immediate_event(AtomEvent(4, None, None, None), is_head=True)
        evt.add_immediate_event(AtomEvent(5, None, None, None), is_head=False)
        evt.add_immediate_event(AtomEvent(6, None, None, None), is_head=False)
        evt.add_immediate_event(AtomEvent(7, None, None, None), is_head=True)
        evt.add_immediate_event(AtomEvent(8, None, None, None), is_head=True)
        self.assertEqual(evt.immediate_event_count, 8)

        iter_evt: Optional[ActualEvent] = evt.immediate_event_head.next_event
        event_ids = []
        while iter_evt is not None:
            event_ids.append(iter_evt.id)
            iter_evt = iter_evt.next_event
        self.assertListEqual(event_ids, [8, 7, 4, 3, 1, 2, 5, 6])

        evt.clear()
        self.assertIsNone(evt.immediate_event_head.next_event)
        self.assertIsNone(evt.immediate_event_tail)
        self.assertEqual(evt.immediate_event_head.next_event, evt.immediate_event_tail)
        self.assertEqual(evt.immediate_event_count, 0)

    def test_event_linked_list(self):
        event_linked_list = EventLinkedList()
        self.assertEqual(len(event_linked_list), 0)
        self.assertListEqual([evt for evt in event_linked_list], [])

        evt_list = []
        for i in range(7):
            evt_list.append(CascadeEvent(i, None, None, None))

        evt_list[0].add_immediate_event(evt_list[3])
        evt_list[0].add_immediate_event(evt_list[4])
        evt_list[0].add_immediate_event(evt_list[5])
        evt_list[0].add_immediate_event(evt_list[6])

        event_linked_list.append_tail(evt_list[1])
        event_linked_list.append_head(evt_list[0])
        event_linked_list.append_tail(evt_list[2])
        self.assertEqual(len(event_linked_list), 3)

        event_ids = [event.id for event in event_linked_list]
        self.assertListEqual(event_ids, [0, 1, 2])

        evt = event_linked_list.front()
        self.assertEqual(evt.id, 0)

        # Test `_clear_finished_events()`
        evt_list[0].state = EventState.FINISHED
        evt = event_linked_list.front()
        self.assertIsInstance(evt, ActualEvent)
        self.assertEqual(evt.id, 3)
        self.assertEqual(len(event_linked_list), 6)

        self.assertListEqual([evt.id for evt in event_linked_list], [3, 4, 5, 6, 1, 2])

        evt_list[3].event_type = MaroEvents.PENDING_DECISION
        evt_list[4].event_type = MaroEvents.PENDING_DECISION
        evt_list[5].event_type = MaroEvents.PENDING_DECISION
        evts = event_linked_list.front()
        self.assertTrue(all(isinstance(evt, ActualEvent) for evt in evts))
        self.assertEqual(len(evts), 3)
        self.assertListEqual([evt.id for evt in evts], [3, 4, 5])
        self.assertListEqual([evt.id for evt in event_linked_list], [3, 4, 5, 6, 1, 2])

        event_linked_list.clear()
        self.assertEqual(len(event_linked_list), 0)
        self.assertListEqual([evt for evt in event_linked_list], [])

    def test_gen_event(self):
        """Test event generating correct"""
        evt = self.eb.gen_atom_event(1, 1, (0, 0))

        # fields should be same as specified
        self.assertEqual(AtomEvent, type(evt))
        self.assertEqual(evt.tick, 1)
        self.assertEqual(evt.event_type, 1)
        self.assertEqual(evt.payload, (0, 0))

        evt = self.eb.gen_cascade_event(2, 2, (1, 1, 1))

        self.assertEqual(CascadeEvent, type(evt))
        self.assertEqual(evt.tick, 2)
        self.assertEqual(evt.event_type, 2)
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
            self.assertEqual(
                1, evt.tick, msg="received event tick should be 1")

            # test event payload
            self.assertTupleEqual(
                (1, 3), evt.payload, msg="received event's payload should be (1, 3)")

        evt = self.eb.gen_atom_event(1, 1, (1, 3))

        self.eb.insert_event(evt)

        self.eb.register_event_handler(1, cb)

        self.eb.execute(1)  # dispatch event

    def test_get_finish_events(self):
        """Test if we can get correct finished events"""

        # no finised at first
        self.assertListEqual([], self.eb.get_finished_events(),
                             msg="finished pool should be empty")

        evt = self.eb.gen_atom_event(1, 1, (1, 3))

        self.eb.insert_event(evt)

        self.eb.execute(1)

        # after dispatching, finish pool should contains 1 object
        self.assertEqual(1, len(self.eb.get_finished_events()),
                         msg="after dispathing, there should 1 object")

    def test_get_pending_events(self):
        """Test if we can get correct pending events"""

        # not pending at first
        self.assertEqual(0, len(self.eb.get_pending_events(1)),
                         msg="pending pool should be empty")

        evt = self.eb.gen_atom_event(1, 1, (1, 3))

        self.eb.insert_event(evt)

        self.assertEqual(1, len(self.eb.get_pending_events(1)),
                         msg="pending pool should contains 1 objects")

    def test_reset(self):
        """Test reset, all internal states should be reset"""
        evt = self.eb.gen_atom_event(1, 1, 1)

        self.eb.insert_event(evt)

        self.eb.reset()

        # reset will not clear the tick (key), just clear the pending pool
        self.assertEqual(len(self.eb._pending_events), 1)

        for tick, pending_pool in self.eb._pending_events.items():
            self.assertEqual(0, len(pending_pool))

        self.assertEqual(len(self.eb._finished_events), 0)

    def test_sub_events(self):

        def cb1(evt):
            self.assertEqual(1, evt.payload)

        def cb2(evt):
            self.assertEqual(2, evt.payload)

        self.eb.register_event_handler(1, cb1)
        self.eb.register_event_handler(2, cb2)

        evt: CascadeEvent = self.eb.gen_cascade_event(1, 1, 1)

        evt.add_immediate_event(self.eb.gen_atom_event(1, 2, 2))

        self.eb.insert_event(evt)

        self.eb.execute(1)

    def test_sub_events_with_decision(self):
        evt1 = self.eb.gen_decision_event(1, (1, 1, 1))
        sub1 = self.eb.gen_decision_event(1, (2, 2, 2))
        sub2 = self.eb.gen_decision_event(1, (3, 3, 3))

        evt1.add_immediate_event(sub1, is_head=True)
        evt1.add_immediate_event(sub2)

        self.eb.insert_event(evt1)

        # sub events will be unfold after parent being processed
        decision_events = self.eb.execute(1)

        # so we will get 1 decision events for 1st time executing
        self.assertEqual(1, len(decision_events))
        self.assertEqual(evt1, decision_events[0])

        # mark decision event as executing to make it process following events
        decision_events[0].state = EventState.FINISHED

        # then there will be 2 additional decision event from sub events
        decision_events = self.eb.execute(1)

        self.assertEqual(2, len(decision_events))
        self.assertEqual(sub1, decision_events[0])
        self.assertEqual(sub2, decision_events[1])


if __name__ == "__main__":
    unittest.main()
