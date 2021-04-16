# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


def get_cls(cls_type, index: dict):
    if isinstance(cls_type, str):
        if cls_type not in index:
            raise KeyError(f"A string sampler_type must be one of {list(cls_type.keys())}.")
        return index[cls_type]

    return cls_type
