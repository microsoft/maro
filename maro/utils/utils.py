# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from pickle import loads, dumps


def clone(obj):
    """Clone an object"""
    return loads(dumps(obj))


class DottableDict(dict):
    """A wrapper to dictionary to make possible to key as property"""
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self


def convert_dottable(natural_dict: dict):
    """Convert a dictionary to DottableDict

    Returns:
        DottableDict: doctable object
    """
    dottable_dict = DottableDict(natural_dict)
    for k, v in natural_dict.items():
        if type(v) is dict:
            v = convert_dottable(v)
            dottable_dict[k] = v
    return dottable_dict
