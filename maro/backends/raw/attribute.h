// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef _MARO_BACKEND_RAW_ATTRIBUTE_
#define _MARO_BACKEND_RAW_ATTRIBUTE_

#include <string>
#include <math.h>

#include "common.h"

using namespace std;

namespace maro
{
  namespace backends
  {
    namespace raw
    {
      // Length of the attribute data.
      const int ATTRIBUTE_DATA_LENGTH = 8;

      /// <summary>
      /// Trait struct to support getter template.
      /// </summary>
      template<typename T>
      struct Attribute_Trait
      {
        typedef T type;
      };

      /// <summary>
      /// Attribute for a node, used to hold all supported data type.
      /// </summary>
      class Attribute
      {
        // Chars to hold all data we supported.
        char _data[ATTRIBUTE_DATA_LENGTH];

        // Type of current attribute, defalut is char
        AttrDataType _type = AttrDataType::ACHAR;

      public:
        // Slot number of list attribute, it will alway be 0 for fixed size attribute.
        SLOT_INDEX slot_number = 0;

        // Constructors
        Attribute() noexcept;
        Attribute(ATTR_CHAR value) noexcept;
        Attribute(ATTR_UCHAR value) noexcept;
        Attribute(ATTR_SHORT value) noexcept;
        Attribute(ATTR_USHORT value) noexcept;
        Attribute(ATTR_INT value) noexcept;
        Attribute(ATTR_UINT value) noexcept;
        Attribute(ATTR_LONG value) noexcept;
        Attribute(ATTR_ULONG value) noexcept;
        Attribute(ATTR_FLOAT value) noexcept;
        Attribute(ATTR_DOUBLE value) noexcept;

        /// <summary>
        /// Get type of current attribute.
        /// </summary>
        AttrDataType get_type() const noexcept;

        /// <summary>
        /// Get value of current attribute.
        /// </summary>
        template<typename T>
        typename Attribute_Trait<T>::type get_value() const noexcept;

        /// <summary>
        /// Cast current value into float, for snapshot querying.
        /// </summary>
        operator QUERY_FLOAT() const;

        // Assignment, copy from another attribute (deep copy).
        Attribute& operator=(const Attribute& attr) noexcept;

        // Setters.
        // NOTE: setters will change its type.
        Attribute& operator=(ATTR_CHAR value) noexcept;
        Attribute& operator=(ATTR_UCHAR value) noexcept;
        Attribute& operator=(ATTR_SHORT value) noexcept;
        Attribute& operator=(ATTR_USHORT value) noexcept;
        Attribute& operator=(ATTR_INT value) noexcept;
        Attribute& operator=(ATTR_UINT value) noexcept;
        Attribute& operator=(ATTR_LONG value) noexcept;
        Attribute& operator=(ATTR_ULONG value) noexcept;
        Attribute& operator=(ATTR_FLOAT value) noexcept;
        Attribute& operator=(ATTR_DOUBLE value) noexcept;

        /// <summary>
        /// Is current value is nan, for float type only.
        /// </summary>
        /// <returns>True if value is nan, or false.</returns>
        bool is_nan() const noexcept;
      };


      /// <summary>
      /// Invalid casting
      /// </summary>
      struct AttributeInvalidDataTypeError : public exception
      {
        const char* what() const noexcept override;
      };
    } // namespace raw
  }   // namespace backends
} // namespace maro

#endif
