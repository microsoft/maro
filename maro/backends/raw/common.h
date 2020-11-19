// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef _MARO_BACKENDS_RAW_COMMON
#define _MARO_BACKENDS_RAW_COMMON

#include <cstdint>
#include <limits>
#include <iostream>

namespace maro
{
  namespace backends
  {
    namespace raw
    {
      // Real data type of attributes.
      using ATTR_BYTE = int8_t;
      using ATTR_SHORT = int16_t;
      using ATTR_INT = int32_t;
      using ATTR_LONG = int64_t;
      using ATTR_FLOAT = float;
      using ATTR_DOUBLE = double;
      using QUERING_FLOAT = float;

      // NOTE: this should sync with maro/backends/backend.pxd
      // Common definitions.
      using INT = int;
      using USHORT = unsigned short;
      using UINT = unsigned int;
      using ULONG = unsigned long long;
      using LONG = long long;
      using SHORT = short;

      using IDENTIFIER = unsigned short;
      using NODE_INDEX = unsigned short; // max 65535
      using SLOT_INDEX = unsigned short;

      // max type of node/attribute we can have in one backend instance
      const IDENTIFIER MAX_IDENTIFIERS = USHRT_MAX;
      const USHORT MAX_SNAPSHOTS = USHRT_MAX;

      /// <summary>
      /// Supported data type for attributes.
      /// </summary>
      enum class AttrDataType : char
      {
        ABYTE,
        ASHORT,
        AINT,
        ALONG,
        AFLOAT,
        ADOUBLE,
      };
    } // namespace raw
  }   // namespace backends
} // namespace maro

#endif
