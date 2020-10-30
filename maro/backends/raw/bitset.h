#ifndef _MARO_BACKENDS_RAW_BITSET
#define _MARO_BACKENDS_RAW_BITSET


#include <vector>

#include "common.h"

using namespace std;

namespace maro
{
  namespace backends
  {
    namespace raw
    {
      const auto BITS_PER_BYTE = 8;
      const auto BITS_PER_MASK = sizeof(ULONG) * BITS_PER_BYTE;

      class Bitset
      {
        /// <summary>
        /// Iterator to go over bitset for target value
        /// </summary>
        class BitsetIterator
        {
        public:
          BitsetIterator(Bitset& bitset, bool target);

          /// <summary>
          /// If reach the end of bitset.
          /// </summary>
          /// <returns>True of reach the end, or false</returns>
          bool end();
        };


        vector<ULONG> _masks;

        ULONG _empties;

        //size of bits
        ULONG _bit_size;

      public:
        Bitset(UINT size);

        /// <summary>
        /// Extend mask with spcified size
        /// </summary>
        /// <param name="size">Size to extend, it should be 64 times</param>
        void extend(UINT size);

        /// <summary>
        /// Invert bits
        /// </summary>
        void invert();

        /// <summary>
        /// reset all bit to specified value
        /// </summary>
        /// <param name="">Value to reset</param>
        void reset(bool value=false);

        /// <summary>
        /// Get value at specified index
        /// </summary>
        /// <param name="index">Index of bit</param>
        /// <returns>True if the bit is 1, or 0</returns>
        bool get(LONG index) const;

        void set(LONG index, bool value);


        /// <summary>
        /// Get number of empty slot
        /// </summary>
        ULONG empties();

        ULONG size();

        UINT mask_size();

        /// <summary>
        /// Get an iterator to go over all empty slots
        /// </summary>
        BitsetIterator* get_empty_slots();
      };
    }
  }
}


#endif // !_MARO_BACKENDS_RAW_BITSET
