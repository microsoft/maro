#ifndef _MARO_BACKENDS_RAW_COMMON
#define _MARO_BACKENDS_RAW_COMMON

#include <stdint.h>


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

      // NOTE: this should sync with maro/backends/backend.pxd
      // Common definitions.
      using INT = int;
      using UINT = unsigned int;
      using ULONG = unsigned long long;

      using IDENTIFIER = unsigned int;
      using NODE_INDEX = unsigned short; // max 65535
      using SLOT_INDEX = unsigned short;

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
    }
  }
}


#endif
