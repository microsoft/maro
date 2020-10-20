#ifndef _MARO_BACKEND_RAW_ATTRIBUTE
#define _MARO_BACKEND_RAW_ATTRIBUTE

#include "common.h"

namespace maro
{
    namespace backends
    {
        namespace raw
        {
            /**
             * @brief Attribute for a node, used to hold all supported data type
            */
            class Attribute
            {
                // data
                union
                {
                    ATTR_BYTE _byte;
                    ATTR_SHORT _short;
                    ATTR_INT _int;
                    ATTR_LONG _long;
                    ATTR_FLOAT _float;
                    ATTR_DOUBLE _double;

                    char _data[8];
                } _data alignas(8){0};

                // data type enum, default is int
                AttrDataType _type{AttrDataType::INT};

            public:
                // constructors
                Attribute() = default;
                Attribute(ATTR_BYTE byte_val);
                Attribute(ATTR_SHORT short_val);
                Attribute(ATTR_INT int_val);
                Attribute(ATTR_LONG long_val);
                Attribute(ATTR_FLOAT float_val);
                Attribute(ATTR_DOUBLE double_val);
            }
        } // namespace raw
    }     // namespace backends
} // namespace maro

#endif