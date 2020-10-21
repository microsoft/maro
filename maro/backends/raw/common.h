#ifndef _MARO_BACKENDS_COMMON
#define _MARO_BACKENDS_COMMON

#include <stdint.h>


namespace maro
{
    namespace backends
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
        using UINT = uint64_t;
        using IDENTIFIER = uint64_t;
        using NODE_INDEX = uint64_t;
        using SLOT_INDEX = uint64_t;

        /**
         * @brief Supported data type for attributes.
        */
        enum class AttrDataType : char
        {
            BYTE,
            SHORT,
            INT,
            LONG,
            FLOAT,
            DOUBLE,
        };
    }
}


#endif