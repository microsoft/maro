# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.s

from datetime import datetime

from maro.data_lib import BinaryReader
from maro.event_buffer import EventBuffer

UNPROECESSED_EVENT = "item_not_bind_with_event"


class EventBindBinaryReader:
    """Binary reader that will generate and insert event that defined in meta into event buffer.
    If items that not match any event type, then they will bind to a predefined event UNPROECESSED_EVENT,
    you can handle this by register an event handler.

    Examples:

        .. code-block:: python

            class MyEvents(Enum):
                Event1 = 'event1'
                Event2 = 'event2'

            event_buffer = EventBuffer()

            # Handle events we defined.
            event_buffer.register_event_handler(MyEvents.Event1, on_event1_occur)
            event_buffer.register_event_handler(MyEvents.Event2, on_event1_occur)

            # Handle item that cannot map to event.
            event_buffer.register_event_handler(UNPROECESSED_EVENT, on_unprocessed_item)

            # Create reader within tick (0, 1000), and events will be mapped to MyEvents type.
            reader = EventBindBinaryReader(MyEvents, event_buffer, path_to_bin, 0, 1000)

            # Read and gen event at tick 0.
            reader.read_items(0)

            def on_event1_occur(evt: Event):
                pass

            def on_event1_occur(evt: Event):
                pass

            def on_unprocessed_item(evt: Event):
                pass

    Args:
        event_cls (type): Event class that will be mapped to.
        event_buffer (EventBuffer): Event buffer that used to generate and insert events.
        binary_file_path (str): Path to binary file to read.
        start_tick (int): Start tick to filter, default is 0.
        end_tick (int): End tick to filter, de fault is 100.
        time_unit (str): Unit of tick, available units are "d", "h", "m", "s".
            different unit will affect the reading result.
        buffer_size (int): In memory buffer size.
        enable_value_adjust (bool): If reader should adjust the value of the fields that marked as adjust-able.
    """
    def __init__(self, event_cls: type, event_buffer: EventBuffer, binary_file_path: str,
                 start_tick: int = 0, end_tick=100, time_unit: str = "s", buffer_size: int = 100,
                 enable_value_adjust: bool = False):

        self._reader = BinaryReader(file_path=binary_file_path,
                                    enable_value_adjust=enable_value_adjust, buffer_size=buffer_size)

        self._event_buffer = event_buffer

        self._start_tick = start_tick
        self._end_tick = end_tick
        self._time_unit = time_unit
        self._event_cls = event_cls

        self._picker = self._reader.items_tick_picker(start_time_offset=self._start_tick,
                                                      end_time_offset=self._end_tick, time_unit=self._time_unit)

        self._init_meta()

    @property
    def start_datetime(self) -> datetime:
        """datetime: Start datetime of this binary file."""
        return self._reader.start_datetime

    @property
    def end_datetime(self) -> datetime:
        """datetime: End datetime of this binary file."""
        return self._reader.end_datetime

    @property
    def header(self) -> tuple:
        """tuple: Header in binary file."""
        return self._reader.header

    def read_items(self, tick: int):
        """Read items by tick and generate related events, then insert them into EventBuffer.

        Args:
            tick(int): Tick to get items, NOTE: the tick must specified sequentially.
        """
        if self._picker:
            for item in self._picker.items(tick):
                self._gen_event_by_item(item, tick)

        return None

    def reset(self):
        """Reset states of reader."""
        self._reader.reset()
        self._picker = self._reader.items_tick_picker(start_time_offset=self._start_tick,
                                                      end_time_offset=self._end_tick, time_unit=self._time_unit)

    def _gen_event_by_item(self, item, tick):
        event_name = None

        if self._event_field_name is None and self._default_event is not None:
            # used default event name to gen event
            event_name = self._event_cls(self._default_event)
        elif self._event_field_name is not None:
            val = getattr(item, self._event_field_name, None)

            event_name = self._event_cls(self._events.get(val, self._default_event))
        else:
            event_name = UNPROECESSED_EVENT

        evt = self._event_buffer.gen_atom_event(tick, event_name, payload=item)

        self._event_buffer.insert_event(evt)

    def _init_meta(self):
        meta = self._reader.meta

        # default event display name
        self._default_event = None

        # value -> display name
        self._events = {}

        for event in meta.events:
            self._events[event.value] = event.display_name

            if meta.default_event_name == event.type_name:
                # match, get save the display name
                self._default_event = event.display_name

        self._event_field_name = meta.event_attr_name
