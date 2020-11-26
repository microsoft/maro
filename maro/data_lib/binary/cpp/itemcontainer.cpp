#include "itemcontainer.h"

namespace maro
{
    namespace datalib
    {
        ItemContainer::ItemContainer()
        {
        }

        void ItemContainer::set_buffer(char *buffer)
        {
            _buffer = buffer;
        }

        ItemContainer::~ItemContainer()
        {
            _buffer = nullptr;
        }

        void ItemContainer::set_offset(UINT offset)
        {
            _offset = offset;
        }

#define Getter(type)                                          \
    template <>                                               \
    type ItemContainer::get<type>(int offset)                 \
    {                                                         \
        type r = 0;                                           \
        memcpy(&r, &_buffer[_offset + offset], sizeof(type)); \
        return r;                                             \
    }

        Getter(char)
        Getter(unsigned char)
        Getter(short)
        Getter(unsigned short)
        Getter(int32_t)
        Getter(uint32_t)
        Getter(LONGLONG)
        Getter(ULONGLONG)
        Getter(float)
        Getter(double)

    } // namespace datalib

} // namespace maro
