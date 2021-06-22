# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


class DynamicObject:
    def __init__(self):
        pass

    def __setitem__(self, key, value):
        if key not in self.__dict__:
            self.__dict__[key] = value

            return
        else:
            pre_value = self.__dict__[key]

            if type(pre_value) == list:
                pre_value.append(value)
            else:
                self.__dict__[key] = [pre_value, value]

    def get(self, key, default=None):
        return self.__dict__.get(key, default)
