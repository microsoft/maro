#ifndef _MARO_BACKEND_RAW_ATTRIBUTE
#define _MARO_BACKEND_RAW_ATTRIBUTE

#include "common.h"

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
        AttrDataType _type{ AttrDataType::INT };

      public:
        // constructors
        Attribute() noexcept = default;
        Attribute(ATTR_BYTE byte_val) noexcept;
        Attribute(ATTR_SHORT short_val) noexcept;
        Attribute(ATTR_INT int_val) noexcept;
        Attribute(ATTR_LONG long_val) noexcept;
        Attribute(ATTR_FLOAT float_val) noexcept;
        Attribute(ATTR_DOUBLE double_val) noexcept;

        // cast function
        operator ATTR_BYTE() const noexcept;
        operator ATTR_SHORT() const noexcept;
        operator ATTR_INT() const noexcept;
        operator ATTR_LONG() const noexcept;
        operator ATTR_FLOAT() const noexcept;
        operator ATTR_DOUBLE() const noexcept;


        // assign
        // NOTE: assign with different data type will change internal data type!
        void operator=(const ATTR_BYTE val) noexcept;
        void operator=(const ATTR_SHORT val) noexcept;
        void operator=(const ATTR_INT val) noexcept;
        void operator=(const ATTR_LONG val) noexcept;
        void operator=(const ATTR_FLOAT val) noexcept;
        void operator=(const ATTR_DOUBLE val) noexcept;
      };
    } // namespace raw
  }     // namespace backends
} // namespace maro

#endif
