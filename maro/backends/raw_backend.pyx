# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#cython: language_level=3
#distutils: language = c++

from maro.backends.backend cimport BackendAbc, SnapshotListAbc, UINT, ULONG, IDENTIFIER, NODE_INDEX, SLOT_INDEX


cdef extern from "raw/common.h" namespace "maro::backends::raw" nogil:
    cdef cppclass AttrDataType:
        pass


cdef extern from "raw/common.h" namespace "maro::backends::raw::AttrDataType" nogil:
    cdef AttrDataType BYTE
    cdef AttrDataType SHORT
    cdef AttrDataType INT
    cdef AttrDataType LONG
    cdef AttrDataType FLOAT
    cdef AttrDataType DOUBLE


cdef extern from "raw/attribute.cpp":
    pass


cdef extern from "raw/attribute.h" namespace "maro::backends::raw" nogil: 
    cdef cppclass Attribute:
        pass


cdef extern from "raw/backend.cpp" namespace "maro::backends::raw" nogil:
    pass


cdef extern from "raw/backend.h" namespace "maro::backends::raw" nogil:
    cdef cppclass Backend:
        pass


cdef class RawBackend(BackendAbc):
    pass

cdef class RawSnapshotList(SnapshotListAbc):
    pass