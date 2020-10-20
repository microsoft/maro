#ifndef _MARO_BACKENDS_COMMON
#define _MARO_BACKENDS_COMMON

#include <stdint.h>


namespace maro
{
    namespace backends
    {
        // Real data type of attributes.
        typedef int8_t ATTR_BYTE;
        typedef int16_t ATTR_SHORT;
        typedef int32_t ATTR_INT;
        typedef int64_t ATTR_LONG;
        typedef float ATTR_FLOAT;
        typedef double ATTR_DOUBLE;

        // NOTE: this should sync with maro/backends/backend.pxd
        // Common definitions.
        typedef uint64_t UINT;
        typedef uint64_t IDENTIFIER;
        typedef uint64_t NODE_INDEX;
        typedef uint64_t SLOT_INDEX;

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