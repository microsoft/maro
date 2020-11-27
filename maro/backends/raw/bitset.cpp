// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "bitset.h"

namespace maro
{
  namespace backends
  {
    namespace raw
    {
      Bitset::BitsetIterateObject::BitsetIterateObject()
      {
      }

      template <typename T>
      inline UINT ceil_to_times(UINT n)
      {
        auto b = sizeof(T) * BITS_PER_BYTE;

        return n % b == 0 ? n / b : (floorl(n / b) + 1);
      }

      Bitset::Bitset()
      {
      }

      Bitset::Bitset(UINT size)
      {
        auto vector_size = ceil_to_times<ULONG>(size);

        _masks.resize(vector_size);

        _bit_size = vector_size * BITS_PER_MASK;
      }

      void Bitset::resize(UINT size)
      {
        auto new_size = ceil_to_times<ULONG>(size);

        _masks.resize(new_size);

        _bit_size = new_size * BITS_PER_MASK;
      }

      void Bitset::reset(bool value)
      {
        auto v = value ? ULONG_MAX : 0ULL;

        memset(&_masks[0], v, _masks.size() * sizeof(ULONG));
      }

      ULONG Bitset::size()
      {
        return _bit_size;
      }

      UINT Bitset::mask_size()
      {
        return _masks.size();
      }

      bool Bitset::get(ULONG index) const
      {
        if (index >= _bit_size)
        {
          throw IndexOutRange();
        }

        ULONG i = floorl(index / BITS_PER_MASK);
        auto offset = index % BITS_PER_MASK;

        auto mask = _masks[i];

        auto target = mask >> offset & 0x1ULL;

        return target == 1;
      }

      void Bitset::set(ULONG index, bool value)
      {
        if (index >= _bit_size)
        {
          throw IndexOutRange();
        }

        ULONG i = floorl(index / BITS_PER_MASK);
        auto offset = index % BITS_PER_MASK;

        if (value)
        {
          // set to 1
          _masks[i] |= 0x1ULL << offset;
        }
        else
        {
          _masks[i] &= ~(0x1ULL << offset);
        }
      }

      Bitset::BitsetIterateObject &Bitset::empty_iter_obj()
      {
        // reset it each time
        _iter_obj._mask_index = 0ULL;
        _iter_obj._mask_offset = 0;

        return _iter_obj;
      }

      bool Bitset::is_end(BitsetIterateObject &iter_obj)
      {
        auto index = _iter_obj._mask_index;
        auto offset = _iter_obj._mask_offset;

        if (offset >= BITS_PER_MASK)
        {
          index++;
          offset = 0;
        }

        // find next mask that has empty slot (0)
        if (offset == 0)
        {
          while (index < _masks.size())
          {
            if (_masks[index] != ULLONG_MAX)
            {
              break;
            }

            index++;
          }
        }

        //
        if (index >= _masks.size())
        {
          return true;
        }

        _iter_obj._mask_index = index;
        _iter_obj._mask_offset = offset;

        return false;
      }

      ULONG Bitset::empty_index(BitsetIterateObject &iter_obj)
      {
        auto index = _iter_obj._mask_index;
        auto offset = _iter_obj._mask_offset;

        auto mask = _masks[index];

        for (auto i = offset; i < BITS_PER_MASK; i++)
        {
          mask = mask >> i;

          // check if last bit is 0
          if ((mask & 0x1ULL) == 0)
          {
            // pointer offset to next one
            _iter_obj._mask_offset = offset + 1;

            return index * BITS_PER_MASK + i;
          }
        }

        return 0;
      }

      const char* IndexOutRange::what() const noexcept
      {
        return "Index of bit flag out of range.";
      }
    } // namespace raw
  }   // namespace backends
} // namespace maro
