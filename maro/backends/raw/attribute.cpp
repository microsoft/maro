// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "attribute.h"

namespace maro
{
  namespace backends
  {
    namespace raw
    {
      Attribute::Attribute() noexcept
      {
        memset(_data, 0, ATTRIBUTE_DATA_LENGTH);
      }

// Macro for all type of constructors.
#define CONSTRUCTOR(data_type, type_name)         \
  Attribute::Attribute(data_type value) noexcept  \
  {                                               \
    memcpy(_data, &value, sizeof(data_type));     \
    _type = type_name;                            \
  }

      CONSTRUCTOR(ATTR_CHAR, AttrDataType::ACHAR)
      CONSTRUCTOR(ATTR_UCHAR, AttrDataType::AUCHAR)
      CONSTRUCTOR(ATTR_SHORT, AttrDataType::ASHORT)
      CONSTRUCTOR(ATTR_USHORT, AttrDataType::AUSHORT)
      CONSTRUCTOR(ATTR_INT, AttrDataType::AINT)
      CONSTRUCTOR(ATTR_UINT, AttrDataType::AUINT)
      CONSTRUCTOR(ATTR_LONG, AttrDataType::ALONG)
      CONSTRUCTOR(ATTR_ULONG, AttrDataType::AULONG)
      CONSTRUCTOR(ATTR_FLOAT, AttrDataType::AFLOAT)
      CONSTRUCTOR(ATTR_DOUBLE, AttrDataType::ADOUBLE)

      AttrDataType Attribute::get_type() const noexcept
      {
        return _type;
      }

      Attribute::operator QUERY_FLOAT() const
      {
        switch (_type)
        {
        case AttrDataType::AUCHAR: { return QUERY_FLOAT(get_value<ATTR_UCHAR>()); }
        case AttrDataType::AUSHORT: { return QUERY_FLOAT(get_value<ATTR_USHORT>()); }
        case AttrDataType::AUINT: { return QUERY_FLOAT(get_value<ATTR_UINT>()); }
        case AttrDataType::AULONG: { return QUERY_FLOAT(get_value<ATTR_ULONG>()); }
        case AttrDataType::ACHAR: { return QUERY_FLOAT(get_value<ATTR_CHAR>()); }
        case AttrDataType::ASHORT: { return QUERY_FLOAT(get_value<ATTR_SHORT>()); }
        case AttrDataType::AINT: { return QUERY_FLOAT(get_value<ATTR_INT>()); }
        case AttrDataType::ALONG: { return QUERY_FLOAT(get_value<ATTR_LONG>()); }
        case AttrDataType::AFLOAT: { return QUERY_FLOAT(get_value<ATTR_FLOAT>()); }
        case AttrDataType::ADOUBLE: { return QUERY_FLOAT(get_value<ATTR_DOUBLE>()); }
        default:
          break;
        }

        throw AttributeInvalidDataTypeError();
      }

      bool Attribute::is_nan() const noexcept
      {
        return _type == AttrDataType::AFLOAT && isnan(get_value<ATTR_FLOAT>());
      }

      template<typename T>
      typename Attribute_Trait<T>::type Attribute::get_value() const noexcept
      {
        T value = T();

        // NOTE: we do not check type here, if the type not match, will get invalid value.
        memcpy(&value, _data, sizeof(T));

        return value;
      }

// Macro for attribute getter template.
#define GETTER(type) template type Attribute::get_value<type>() const noexcept;

      GETTER(ATTR_CHAR)
      GETTER(ATTR_UCHAR)
      GETTER(ATTR_SHORT)
      GETTER(ATTR_USHORT)
      GETTER(ATTR_INT)
      GETTER(ATTR_UINT)
      GETTER(ATTR_LONG)
      GETTER(ATTR_ULONG)
      GETTER(ATTR_FLOAT)
      GETTER(ATTR_DOUBLE)

      Attribute& Attribute::operator=(const Attribute& attr) noexcept
      {
        if (this != &attr)
        {
          _type = attr._type;

          memcpy(_data, attr._data, ATTRIBUTE_DATA_LENGTH);
        }

        return *this;
      }

// Macro for setters.
#define SETTER(data_type, value_type)                         \
  Attribute& Attribute::operator=(data_type value) noexcept   \
  {                                                           \
    memcpy(_data, &value, sizeof(data_type));                 \
    _type = value_type;                                       \
    return *this;                                             \
  }

      SETTER(ATTR_CHAR, AttrDataType::ACHAR)
      SETTER(ATTR_UCHAR, AttrDataType::AUCHAR)
      SETTER(ATTR_SHORT, AttrDataType::ASHORT)
      SETTER(ATTR_USHORT, AttrDataType::AUSHORT)
      SETTER(ATTR_INT, AttrDataType::AINT)
      SETTER(ATTR_UINT, AttrDataType::AUINT)
      SETTER(ATTR_LONG, AttrDataType::ALONG)
      SETTER(ATTR_ULONG, AttrDataType::AULONG)
      SETTER(ATTR_FLOAT, AttrDataType::AFLOAT)
      SETTER(ATTR_DOUBLE, AttrDataType::ADOUBLE)


      const char* AttributeInvalidDataTypeError::what() const noexcept
      {
        return "Invalid attribute data type.";
      }
    } // namespace raw
  }   // namespace backends
} // namespace maro
