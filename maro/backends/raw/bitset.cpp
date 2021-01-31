// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "bitset.h"

namespace maro
{
  namespace backends
  {
    namespace raw
    {
      inline size_t ceil_to_times(UINT number)
      {
        auto bits = sizeof(ULONG) * BITS_PER_BYTE;

        return number % bits == 0 ? number / bits : (floorl(number / bits) + 1);
      }

      Bitset::Bitset()
      {
      }

      Bitset::Bitset(UINT size)
      {
        auto vector_size = ceil_to_times(size);

        _masks.resize(vector_size);

        _bit_size = ULONG(vector_size) * BITS_PER_MASK;
      }

      Bitset& Bitset::operator=(const Bitset& set) noexcept
      {
        if (this != &set)
        {
          _masks.resize(set._masks.size());

          memcpy(&_masks[0], &set._masks[0], _masks.size() * sizeof(ULONG));

          _bit_size = set._bit_size;
        }

        return *this;
      }

      void Bitset::resize(UINT size) noexcept
      {
        auto new_size = ceil_to_times(size);

        _masks.resize(new_size);

        _bit_size = ULONG(new_size) * BITS_PER_MASK;
      }

      void Bitset::reset(bool value) noexcept
      {
        auto v = value ? ULONG_MAX : 0ULL;

        memset(&_masks[0], v, _masks.size() * sizeof(ULONG));
      }

      ULONG Bitset::size() const noexcept
      {
        return _bit_size;
      }

      UINT Bitset::mask_size() const noexcept
      {
        return _masks.size();
      }

      bool Bitset::get(ULONG index) const noexcept
      {
        if (index >= _bit_size)
        {
          return false;
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
          throw BitsetIndexOutRangeError();
        }

        ULONG i = floorl(index / BITS_PER_MASK);
        auto offset = index % BITS_PER_MASK;

        if (value)
        {
          // Set to 1.
          _masks[i] |= 0x1ULL << offset;
        }
        else
        {
          _masks[i] &= ~(0x1ULL << offset);
        }
      }


      const char* BitsetIndexOutRangeError::what() const noexcept
      {
        return "Index of bit flag out of range.";
      }
    } // namespace raw
  }   // namespace backends
} // namespace maro
