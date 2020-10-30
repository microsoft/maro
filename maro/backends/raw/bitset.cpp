#include "bitset.h"


namespace maro
{
  namespace backends
  {
    namespace raw
    {
      Bitset::BitsetIterator::BitsetIterator(Bitset& bitset, bool target)
      {

      }


      bool Bitset::BitsetIterator::end()
      {
        return true;
      }

      template<typename T>
      inline UINT ceil_to_times(UINT n)
      {
        auto b = sizeof(T) * BITS_PER_BYTE;

        return n % b == 0 ? n/b : (floorl(n / b) + 1);
      }


      Bitset::Bitset(UINT size)
      {
        auto vector_size = ceil_to_times<ULONG>(size);

        _masks.resize(vector_size);

        _bit_size = vector_size * BITS_PER_MASK;

        _empties = _bit_size;
      }

      void Bitset::extend(UINT size)
      {
        auto new_size = ceil_to_times<ULONG>(size);

        _masks.resize(_masks.size() + new_size);

        _bit_size += new_size * BITS_PER_MASK;

        _empties += new_size * BITS_PER_MASK;
      }

      void Bitset::invert()
      {

      }

      void Bitset::reset(bool value)
      {
        auto v = value ? 1 : 0;

        memset(&_masks[0], v, _masks.size() * sizeof(ULONG));
      }

      ULONG Bitset::empties()
      {
        return _empties;
      }

      ULONG Bitset::size()
      {
        return _bit_size;
      }

      UINT Bitset::mask_size()
      {
        return _mask.size();
      }

      Bitset::BitsetIterator* Bitset::get_empty_slots()
      {
        return nullptr;
      }

      bool Bitset::get(LONG index) const
      {
        ULONG i = floorl(index / sizeof(ULONG));
        auto offset = i % sizeof(ULONG);

        auto mask = _masks[i];

        auto target = mask >> offset & 0x1;

        return target == 1;
      }

      void Bitset::set(LONG index, bool value)
      {
        ULONG i = floorl(index / 64);
        auto offset = i % sizeof(ULONG);

        if (value)
        {
          // set to 1
          _masks[i] |= 0x1ULL << offset;
        }
        else
        {
          _masks[i] &= !(0x1ULL << offset);
        }
      }
    }
  }
}
