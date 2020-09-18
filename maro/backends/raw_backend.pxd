# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#cython: language_level=3


# TODO: another implementation with c/c++ to support more features

from maro.backends.backend cimport BackendAbc

cdef class RawBackend(BackendAbc):
    pass