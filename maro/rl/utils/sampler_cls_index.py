# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl.storage import UniformSampler

SAMPLER = {
    "uniform": UniformSampler,
}

def get_sampler_cls(sampler_type):
    if isinstance(sampler_type, str):
        if sampler_type not in SAMPLER:
            raise KeyError(f"A string sampler_type must be one of {list(SAMPLER.keys())}.")
        return SAMPLER[sampler_type]

    return sampler_type
