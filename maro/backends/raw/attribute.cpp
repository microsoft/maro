// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "attribute.h"

namespace maro
{
  namespace backends
  {
    namespace raw
    {

#define CONSTRUCTOR(m_type, m_field, m_internal_type) \
  Attribute::Attribute(m_type val) noexcept           \
  {                                                   \
    _type = m_internal_type;                          \
    _data.m_field = val;                              \
  }

#define DATA_ASSIGN(m_type, m_field, m_internal_type) \
  void Attribute::operator=(m_type val)               \
  {                                                   \
    _type = m_internal_type;                          \
    _data.m_field = val;                              \
  }

      // Before assign any value, all field will be 0, so we do care about data type here, even data type not match
#define DATA_GETTER(m_name, m_field, m_type, m_internal_type) \
  m_type Attribute::get_##m_name()                            \
  {                                                           \
    return _data.m_field;                                     \
  }

      CONSTRUCTOR(ATTR_BYTE, _byte, AttrDataType::ABYTE)
      CONSTRUCTOR(ATTR_DOUBLE, _double, AttrDataType::ADOUBLE)
      CONSTRUCTOR(ATTR_FLOAT, _float, AttrDataType::AFLOAT)
      CONSTRUCTOR(ATTR_INT, _int, AttrDataType::AINT)
      CONSTRUCTOR(ATTR_LONG, _long, AttrDataType::ALONG)
      CONSTRUCTOR(ATTR_SHORT, _short, AttrDataType::ASHORT)

      DATA_ASSIGN(ATTR_BYTE, _byte, AttrDataType::ABYTE)
      DATA_ASSIGN(ATTR_SHORT, _short, AttrDataType::ASHORT)
      DATA_ASSIGN(ATTR_INT, _int, AttrDataType::AINT)
      DATA_ASSIGN(ATTR_LONG, _long, AttrDataType::ALONG)
      DATA_ASSIGN(ATTR_FLOAT, _float, AttrDataType::AFLOAT)
      DATA_ASSIGN(ATTR_DOUBLE, _double, AttrDataType::ADOUBLE)

      DATA_GETTER(byte, _byte, ATTR_BYTE, AttrDataType::ABYTE)
      DATA_GETTER(short, _short, ATTR_SHORT, AttrDataType::ASHORT)
      DATA_GETTER(int, _int, ATTR_INT, AttrDataType::AINT)
      DATA_GETTER(long, _long, ATTR_LONG, AttrDataType::ALONG)
      DATA_GETTER(float, _float, ATTR_FLOAT, AttrDataType::AFLOAT)
      DATA_GETTER(double, _double, ATTR_DOUBLE, AttrDataType::ADOUBLE)

      Attribute::operator ATTR_FLOAT()
      {
        switch (_type)
        {
        case AttrDataType::ABYTE:
          return ATTR_FLOAT(_data._byte);
          break;
        case AttrDataType::ASHORT:
          return ATTR_FLOAT(_data._short);
          break;
        case AttrDataType::AINT:
          return ATTR_FLOAT(_data._int);
          break;
        case AttrDataType::ALONG:
          return ATTR_FLOAT(_data._long);
          break;
        case AttrDataType::AFLOAT:
          return _data._float;
          break;
        case AttrDataType::ADOUBLE:
          return ATTR_FLOAT(_data._double);
          break;
        default:
          break;
        }

        throw InvalidOperation();
      }
      void Attribute::operator=(const Attribute &attr)
      {
        _type = attr._type;

        memcpy(_data._data, attr._data._data, sizeof(char) * 8);
      }

      bool Attribute::is_nan()
      {
        return _type == AttrDataType::AFLOAT && isnan(_data._float);
      }
    } // namespace raw
  }   // namespace backends
} // namespace maro
