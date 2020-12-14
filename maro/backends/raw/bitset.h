// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef _MARO_BACKENDS_RAW_BITSET_
#define _MARO_BACKENDS_RAW_BITSET_

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
      /// A simple bitset implementation.
      /// </summary>
      class Bitset
      {
        // Masks of current bitset, we use ULL for each item.
        vector<ULONG> _masks;

        // Size of bits.
        ULONG _bit_size = 0;
      public:
        Bitset();
        Bitset(UINT size);

        // Copy all from input set.
        Bitset& operator=(const Bitset& set) noexcept;

        /// <summary>
        /// Resize bitset with spcified size.
        /// </summary>
        /// <param name="size">Size to extend, it should be 64 times.</param>
        void resize(UINT size) noexcept;

        /// <summary>
        /// Reset all bit to specified value.
        /// </summary>
        /// <param name="">Value to reset.</param>
        void reset(bool value = false) noexcept;

        /// <summary>
        /// Get value at specified index.
        /// </summary>
        /// <param name="index">Index of bit.</param>
        /// <returns>True if the bit is 1, or false for 0 (not exist).</returns>
        bool get(ULONG index) const noexcept;

        /// <summary>
        /// Set value for specified position.
        /// </summary>
        /// <param name="index">Index of item.</param>
        /// <param name="value">Value to set.</param>
        void set(ULONG index, bool value);

        /// <summary>
        /// Current size of items (in bit).
        /// </summary>
        /// <returns>Number of bits.</returns>
        ULONG size() const noexcept;

        /// <summary>
        /// Get size of mask items (in ULL).
        /// </summary>
        /// <returns>Number of mask items.</returns>
        UINT mask_size() const noexcept;
      };


      /// <summary>
      /// Query index out of range.
      /// </summary>
      struct BitsetIndexOutRangeError : public exception
      {
        const char* what() const noexcept override;
      };
    } // namespace raw
  }   // namespace backends
} // namespace maro

#endif // !_MARO_BACKENDS_RAW_BITSET_
