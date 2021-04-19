# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .sampler import UniformSampler

SAMPLER_CLS = {
    "uniform": UniformSampler,
}
