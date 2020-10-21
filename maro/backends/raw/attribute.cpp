#include "attribute.h"

namespace maro
{
  namespace backends
  {
    namespace raw
    {

#define CONSTRUCTOR(m_type, m_field, m_internal_type) \
    Attribute::Attribute(##m_type val) noexcept       \
    {                                                 \
        _type = m_internal_type;                      \
        _data.##m_field = val;                        \
    }

#define CAST_OPERATOR(m_operator, m_field)           \
    Attribute::operator m_operator() const noexcept  \
    {                                                \
        return _data.m_field;                        \
    }

#define DATA_ASSIGN(m_type, m_field, m_internal_type) \
    void Attribute::operator =(##m_type val) noexcept \
    {                                                 \
        _data.##m_field = val;                        \
        _type = m_internal_type;                      \
    }
        

      CONSTRUCTOR(ATTR_BYTE, _byte, AttrDataType::BYTE)
      CONSTRUCTOR(ATTR_DOUBLE, _double, AttrDataType::DOUBLE)
      CONSTRUCTOR(ATTR_FLOAT, _float, AttrDataType::FLOAT)
      CONSTRUCTOR(ATTR_INT, _int, AttrDataType::INT)
      CONSTRUCTOR(ATTR_LONG, _long, AttrDataType::LONG)
      CONSTRUCTOR(ATTR_SHORT, _short, AttrDataType::SHORT)


      CAST_OPERATOR(ATTR_BYTE, _byte)
      CAST_OPERATOR(ATTR_SHORT, _short)
      CAST_OPERATOR(ATTR_INT, _int)
      CAST_OPERATOR(ATTR_LONG, _long)


      Attribute::operator ATTR_FLOAT() const noexcept
      {
        if (_type == AttrDataType::FLOAT || _type == AttrDataType::DOUBLE)
        {
          return _data._float;
        }
        else {
          return ATTR_FLOAT(_data._long);
        }
      }

      Attribute::operator ATTR_DOUBLE() const noexcept
      {
        if (_type == AttrDataType::FLOAT || _type == AttrDataType::DOUBLE)
        {
          return _data._double;
        }
        else
        {
          return ATTR_DOUBLE(_data._long);
        }
      }

      DATA_ASSIGN(ATTR_BYTE, _byte, AttrDataType::BYTE)
      DATA_ASSIGN(ATTR_SHORT, _short, AttrDataType::SHORT)
      DATA_ASSIGN(ATTR_INT, _int, AttrDataType::INT)
      DATA_ASSIGN(ATTR_LONG, _long, AttrDataType::LONG)
      DATA_ASSIGN(ATTR_FLOAT, _float, AttrDataType::FLOAT)
      DATA_ASSIGN(ATTR_DOUBLE, _double, AttrDataType::DOUBLE)

    }
  }
}
