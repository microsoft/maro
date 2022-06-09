// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef _MARO_BACKEND_RAW_COMMON_
#define _MARO_BACKEND_RAW_COMMON_

#include <cstdint>

using namespace std;

namespace maro
{
  namespace backends
  {
    namespace raw
    {
      using UCHAR = unsigned char;
      using USHORT = unsigned short;
      using UINT = uint32_t;
      using LONG = long long;
      using ULONG = unsigned long long;

      using NODE_TYPE = unsigned short;
      using ATTR_TYPE = uint32_t;

      const size_t MAX_NODE_TYPE = USHRT_MAX;
      const size_t MAX_ATTR_TYPE = USHRT_MAX;
      const size_t MAX_SLOT_NUMBER = UINT32_MAX;

      using NODE_INDEX = uint32_t;
      using SLOT_INDEX = uint32_t;
      using QUERY_FLOAT = double;  // TODO: Precision issue for Long data type.

      using ATTR_CHAR = char;
      using ATTR_UCHAR = unsigned char;
      using ATTR_SHORT = short;
      using ATTR_USHORT = unsigned short;
      using ATTR_INT = int32_t;
      using ATTR_UINT = uint32_t;
      using ATTR_LONG = int64_t;
      using ATTR_ULONG = uint64_t;
      using ATTR_FLOAT = float;
      using ATTR_DOUBLE = double;


      /// <summary>
      /// Attribute data type.
      /// </summary>
      enum class AttrDataType : char
      {
        ACHAR,
        AUCHAR,
        ASHORT,
        AUSHORT,
        AINT,
        AUINT,
        ALONG,
        AULONG,
        AFLOAT,
        ADOUBLE,
        APOINTER,
      };
    }
  }
}

#endif // ! _MARO_BACKEND_RAW_COMMON_
