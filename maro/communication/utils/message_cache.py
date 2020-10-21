# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict, abc


class MessageCache(abc.MutableMapping):
    def __init__(self, max_length):
        self._max_length = max_length
        self._dict = defaultdict(list)

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, value):
        if isinstance(value, list) and len(value) > self._max_length:
            value = value[-self._max_length:]
        self._dict[key] = value

    def __delitem__(self, key):
        del self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __repr__(self):
        return str(self._dict)

    def append(self, key, value):
        if len(self._dict[key]) == self._max_length:
            self._dict[key].pop(0)

        self._dict[key].append(value)
