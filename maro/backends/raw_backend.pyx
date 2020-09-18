# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#cython: language_level=3


from maro.backends.backend cimport BackendAbc


cdef class RawBackend(BackendAbc):
    pass