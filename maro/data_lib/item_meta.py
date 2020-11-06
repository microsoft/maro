# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import random
import re
import warnings
from collections import namedtuple
from struct import Struct
from typing import List, Union

from yaml import SafeDumper, SafeLoader, YAMLObject, safe_dump, safe_load

from maro.data_lib.common import dtype_pack_map
from maro.utils.exception.data_lib_exeption import MetaTimestampNotExist


class EntityAttr(YAMLObject):
    """Entity attribute in yaml."""
    yaml_tag = u"!MaroAttribute"
    yaml_loader = SafeLoader
    yaml_dumper = SafeDumper

    def __init__(self, name, dtype: str, slot: int, raw_name: str, adjust_ratio: tuple, tzone=None):
        self.name = name
        self.dtype = dtype
        self.slot = slot
        self.raw_name = raw_name
        self.adjust_ratio = adjust_ratio
        self.tzone = None


class Event(YAMLObject):
    """Event from yaml."""
    yaml_tag = u"!MaroEvent"
    yaml_loader = SafeLoader
    yaml_dumper = SafeDumper

    def __init__(self, display_name: str, type_name: str, value: object):
        self.display_name = display_name
        self.type_name = type_name
        self.value = value


class BinaryMeta:
    """Meta for binary file."""

    def __init__(self):
        self._item_nt: namedtuple = None
        self._item_struct: Struct = None

        self._tzone = None
        # which attribute used as events

        self._event_attr_name = None
        # if value cannot matched to any event definition, then treat it as default
        self._default_event_name = None
        self._raw_cols = []
        # fields need adjust
        self._adjust_attrs = {}
        # EntityAttr
        self._attrs: List[EntityAttr] = []
        self._events: List[Event] = []

    @property
    def events(self) -> List[Event]:
        """List[Event]: Events definition."""
        return self._events

    @property
    def default_event_name(self) -> str:
        """str: Default event name, if no value matched."""
        return self._default_event_name

    @property
    def event_attr_name(self) -> str:
        """str: Event attribute name."""
        return self._event_attr_name

    @property
    def time_zone(self):
        """Time zone of this meta, used to correct timestamp."""
        return self._tzone

    @property
    def item_size(self) -> int:
        """int: Item binary size (in bytes)."""
        return self._item_struct.size

    @property
    def columns(self) -> dict:
        """dict: Columns to extract."""
        return {a.name: a.raw_name for a in self._attrs}

    def items(self) -> dict:
        """dict: Attribute items."""
        return {a.name: a.dtype for a in self._attrs}

    def from_file(self, file: str):
        """Read meta from yaml file."""
        assert os.path.exists(file)

        with open(file, "rt") as fp:
            conf = safe_load(fp)

            self._validate(conf)

            self._build_item_struct()

    def from_bytes(self, meta_bytes: Union[bytes, bytearray, memoryview]):
        """Construct meta from bytes.

        Args:
            meta_bytes (Union[bytes, bytearray, memoryview]): Bytes content of meta.
        """
        assert meta_bytes is not None

        self._events.clear()
        self._attrs.clear()
        self._raw_cols.clear()

        conf = safe_load(meta_bytes[:].decode())

        # NOTE: this methods used to load meta bytes from converted binary,
        # which is our processed format, so we do not need validate here
        self._events.extend(conf.get("events", []))

        self._attrs.extend(conf.get("attributes", []))

        self._event_attr_name = conf.get("event_attr_name", None)
        self._default_event_name = conf.get("default_event_name", None)

        self._raw_cols = [(a.raw_name, a.dtype) for a in self._attrs]

        self._adjust_attrs = {i: a.adjust_ratio for i, a, in enumerate(
            self._attrs) if a.adjust_ratio is not None}

        self._build_item_struct()

    def from_dict(self, meta_dict: dict):
        """Construct meta from dictionary.

        Args:
            meta_dict (dict): Meta dictionary.
        """
        self._validate(meta_dict)

        self._build_item_struct()

    def to_bytes(self):
        """Convert meta into bytes.

        Returns:
            bytes: Bytes content of current meta.
        """
        return safe_dump(
            {
                "events": self._events,
                "attributes": self._attrs,
                "default_event_name": self._default_event_name,
                "event_attr_name": self._event_attr_name
            },
        ).encode()

    def get_item_values(self, row: dict) -> Union[list, tuple]:
        """Retrieve value for item.

        Args:
            row (dict): A row that from a csv file.

        Returns:
            Union[list, tuple]: Get value for configured attributes from dict.
        """
        # NOTE: keep the order
        return (row[col] for col in self._raw_cols)

    def item_to_bytes(self, item_values: Union[tuple, list], out_bytes: Union[memoryview, bytearray]) -> int:
        """Convert item into bytes.

        Args:
            item_values (Union[tuple, list]): Value of attributes used to construct item.
            out_bytes (Union[memoryview, bytearray]): Item bytes content.

        Returns:
            int: Result item size.
        """
        self._item_struct.pack_into(out_bytes, 0, *item_values)

        return self.item_size

    def item_from_bytes(self, item_bytes: Union[bytes, bytearray, memoryview], adjust_value: bool = False):
        """Convert bytes into item (namedtuple).

        Args:
            item_bytes (Union[bytes, bytearray, memoryview]): Item byte content.
            adjust_value (bool): If need to adjust value for attributes that enabled this feature.

        Returns:
            namedtuple: Result item tuple.
        """
        item_tuple = self._item_struct.unpack_from(item_bytes, 0)

        if adjust_value:
            # convert it into list to that we can change the value
            item_tuple = list(item_tuple)

            for index, ratio in self._adjust_attrs.items():
                # make it percentage
                item_tuple[index] += random.randrange(int(ratio[0]), int(ratio[1])) * 0.01 * item_tuple[index]

        return self._item_nt._make(item_tuple)

    def _build_item_struct(self):
        """Build item struct use field name in meta."""
        self._item_nt = namedtuple("Item", [a.name for a in self._attrs])

        fmt: str = "<" + "".join([dtype_pack_map[a.dtype] for a in self._attrs])

        self._item_struct = Struct(fmt)

    def _validate(self, conf: dict):
        # attributes
        attributes_def = conf.get("entity", {})

        # special events
        self._event_attr_name = attributes_def.get("_event", None)

        has_timestamp_attr = False

        for attr_name, attr_settings in attributes_def.items():
            if type(attr_settings) != dict:
                continue

            dtype = attr_settings.get("dtype", "i")
            col_name = attr_settings.get("column", None)
            tzone = attr_settings.get("tzone", None)
            slots = attr_settings.get("slot", 1)
            adjust_ratio = attr_settings.get("adjust_ratio")

            if dtype in dtype_pack_map and re.match(r"^[a-z A-Z]+", attr_name):
                # keep the index, as we need to change the value later
                if adjust_ratio is not None:
                    self._adjust_attrs[len(self._attrs)] = adjust_ratio

                # TODO: raise error instead
                attr = EntityAttr(attr_name, dtype, slots, col_name, adjust_ratio, tzone)

                if attr_name == "timestamp":
                    has_timestamp_attr = True
                    dtype = "i8"
                    # keep the timestamp as first field
                    self._attrs.insert(0, attr)
                    self._raw_cols.insert(0, attr)
                else:
                    self._attrs.append(attr)
                    self._raw_cols.append((col_name, dtype))
            else:
                warnings.warn(f"invalid attribute {attr_name}, ignore it")

        if not has_timestamp_attr:
            raise MetaTimestampNotExist()

        # events
        events_def = conf.get("events", {})

        self._default_event_name = events_def.get("_default", None)

        for evt_type_name, event_settings in events_def.items():
            if type(event_settings) != dict:
                continue

            display_name = event_settings.get("display_name", evt_type_name)
            value_in_csv = event_settings.get("value_in_csv", None)

            self._events.append(
                Event(display_name, evt_type_name, value_in_csv))


__all__ = ["BinaryMeta"]
