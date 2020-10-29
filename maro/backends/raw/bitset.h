#ifndef _MARO_BACKENDS_RAW_BITSET
#define _MARO_BACKENDS_RAW_BITSET


#include <vector>


using namespace std;

namespace maro
{
  namespace backends
  {
    namespace raw
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

      class Bitset
      {
        vector<ULONG> _masks;

        size_t _empties;

      public:
        Bitset(UINT size);

        /// <summary>
        /// Expend mask with spcified size
        /// </summary>
        /// <param name="size">Size to expend, it should be 64 times</param>
        void expend(UINT size);

        /// <summary>
        /// Invert bits
        /// </summary>
        void invert();

        /// <summary>
        /// reset all bit to specified value
        /// </summary>
        /// <param name="">Value to reset</param>
        void reset(bool value);

        /// <summary>
        /// Get value at specified index
        /// </summary>
        /// <param name="index">Index of bit</param>
        /// <returns>True if the bit is 1, or 0</returns>
        bool operator[](size_t index) const;


        /// <summary>
        /// Get number of empty slot
        /// </summary>
        void empties();

        /// <summary>
        /// Get an iterator to go over all empty slots
        /// </summary>
        BitsetIterator& get_empty_slots();
      };
    }
  }
}


#endif // !_MARO_BACKENDS_RAW_BITSET
