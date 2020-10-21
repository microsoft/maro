# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#cython: language_level=3
#distutils: language = c++

from maro.backends.backend cimport BackendAbc, SnapshotListAbc, UINT, ULONG, IDENTIFIER, NODE_INDEX, SLOT_INDEX

cdef class RawBackend(BackendAbc):
    pass


cdef class RawSnapshotList(SnapshotListAbc):
    pass