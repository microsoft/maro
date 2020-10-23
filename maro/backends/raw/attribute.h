#ifndef _MARO_BACKEND_RAW_ATTRIBUTE
#define _MARO_BACKEND_RAW_ATTRIBUTE

#include "common.h"

using namespace std;

namespace maro
{
  namespace backends
  {
    namespace raw
    {
      /// <summary>
      /// Attribute for a node, used to hold all supported data type
      /// </summary>
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
        } _data alignas(8) { 0 };

        // data type enum, default is int
        AttrDataType _type{ AttrDataType::ABYTE };

      public:
        // constructors
        Attribute() noexcept = default;
        Attribute(ATTR_BYTE byte_val) noexcept;
        Attribute(ATTR_SHORT short_val) noexcept;
        Attribute(ATTR_INT int_val) noexcept;
        Attribute(ATTR_LONG long_val) noexcept;
        Attribute(ATTR_FLOAT float_val) noexcept;
        Attribute(ATTR_DOUBLE double_val) noexcept;

        // getters
        ATTR_BYTE get_byte();
        ATTR_SHORT get_short();
        ATTR_INT get_int();
        ATTR_LONG get_long();
        ATTR_FLOAT get_float();
        ATTR_DOUBLE get_double();

        // setters
        // NOTE: these setters will change inernal data type
        void operator=(const ATTR_BYTE val);
        void operator=(const ATTR_SHORT val);
        void operator=(const ATTR_INT val);
        void operator=(const ATTR_LONG val);
        void operator=(const ATTR_FLOAT val);
        void operator=(const ATTR_DOUBLE val);

        /// <summary>
        /// Used to cast current data to float, for quering result
        /// </summary>
        operator ATTR_FLOAT();
      };
    } // namespace raw
  }     // namespace backends
} // namespace maro

#endif
