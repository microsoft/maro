#ifndef _MARO_BACKENDS_RAW_BITSET
#define _MARO_BACKENDS_RAW_BITSET

#include <memory>
#include <vector>

#include "common.h"

using namespace std;

namespace maro
{
  namespace backends
  {
    namespace raw
    {
      const USHORT BITS_PER_BYTE = 8;
      const USHORT BITS_PER_MASK = sizeof(ULONG) * BITS_PER_BYTE;

      /// <summary>
      /// Query index out of range
      /// </summary>
      class IndexOutRange : public exception
      {
      };

      class Bitset
      {
        /// <summary>
        /// Iterator to go over bitset for target value
        /// </summary>
        class BitsetIterateObject
        {
          friend class Bitset;

          ULONG _mask_index{0};
          // when offset == 0, iterator will check if current mask equals to MAX_ULONG
          USHORT _mask_offset{0};

        public:
          BitsetIterateObject();
        };

        vector<ULONG> _masks;

        //size of bits
        ULONG _bit_size{0};

        BitsetIterateObject _iter_obj;

      public:
        Bitset();
        Bitset(UINT size);

        /// <summary>
        /// Extend mask with spcified size
        /// </summary>
        /// <param name="size">Size to extend, it should be 64 times</param>
        void resize(UINT size);

        /// <summary>
        /// reset all bit to specified value
        /// </summary>
        /// <param name="">Value to reset</param>
        void reset(bool value = false);

        /// <summary>
        /// Get value at specified index
        /// </summary>
        /// <param name="index">Index of bit</param>
        /// <returns>True if the bit is 1, or 0</returns>
        bool get(ULONG index) const;

        /// <summary>
        /// Set value for specified position
        /// </summary>
        /// <param name="index">Index of item</param>
        /// <param name="value">Value to set</param>
        void set(ULONG index, bool value);

        /// <summary>
        /// Current size of items (in bit)
        /// </summary>
        /// <returns>Number of bits</returns>
        ULONG size();

        /// <summary>
        /// Get size of mask items (in ULL)
        /// </summary>
        /// <returns>Number of mask items</returns>
        UINT mask_size();

        /// <summary>
        /// Get an iterator to go over all empty slots
        /// </summary>
        BitsetIterateObject &empty_iter_obj();

        /// <summary>
        /// If reach the end
        /// </summary>
        /// <param name="iter_obj"></param>
        bool is_end(BitsetIterateObject &iter_obj);

        /// <summary>
        /// Get next empty index
        /// </summary>
        ULONG empty_index(BitsetIterateObject &iter_obj);
      };
    } // namespace raw
  }   // namespace backends
} // namespace maro

#endif // !_MARO_BACKENDS_RAW_BITSET
