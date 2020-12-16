# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import warnings
from datetime import datetime

from dateutil.relativedelta import relativedelta
from dateutil.tz import UTC

timestamp_start = datetime(1970, 1, 1, 0, 0, 0, tzinfo=UTC)


def utc_timestamp_to_timezone(timestamp: int, timezone):
    """Convert utc timestamp into specified tiemzone datetime.

    Args:
        timestamp(int): UTC timestamp to convert.
        timezone: Target timezone.
    """
    if sys.platform == "win32":
        # windows do not support negative timestamp, use this to support it
        return (timestamp_start + relativedelta(seconds=timestamp)).astimezone(timezone)
    else:
        return datetime.utcfromtimestamp(timestamp).replace(tzinfo=UTC).astimezone(timezone)


class DocableDict:
    """A thin wrapper that provide a read-only dictionary with customized doc.

    Args:
        doc (str): Customized doc of the dict.
        kwargs (dict): Dictionary items to store.
    """

    def __init__(self, doc: str, **kwargs):
        self._original_dict = kwargs
        DocableDict.__doc__ = doc

    def __getattr__(self, name):
        return getattr(self._original_dict, name, None)

    def __getitem__(self, k):
        return self._original_dict[k]

    def __setitem__(self, k, v):
        warnings.warn("Do not support add new key")

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self):
        return self._original_dict.__repr__()

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self._original_dict)
