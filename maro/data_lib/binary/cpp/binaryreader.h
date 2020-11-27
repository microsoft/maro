// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef _MARO_DATALIB_BINARYREADER_
#define _MARO_DATALIB_BINARYREADER_

#include <unordered_map>
#include <iostream>
#include <fstream>
#include <string>

#include "common.h"
#include "itemcontainer.h"

using namespace std;

namespace maro
{
  namespace datalib
  {
    const ULONGLONG INVALID_FILTER = 0ULL;

    class BinaryReader
    {
    private:
      // binary file header
      BinHeader _header;

      // binary meta
      Meta _meta;

      // file handler we are reading
      ifstream _file;

      // if file opened
      bool _is_opened{ false };

      // buffer to read from binary file
      char _buffer[BUFFER_LENGTH];

      // container to hold current item
      ItemContainer _item;

      // max items in binary buffer, it is times of item size
      long max_items_in_buffer{ 0 };

      // current item index in binary buffer, used to calc offset
      int cur_item_index{ -1 };

      // offset of data part, used for reset
      streampos _data_offset{ 0 };

      // if filter enabled
      bool _is_filter_enabled{ false };

      // start timestamp to filter (included in result)
      ULONGLONG _filter_start{ INVALID_FILTER };

      // end timestamp to filter (exclude in result)
      ULONGLONG _filter_end{ INVALID_FILTER };

      // used to save the offset in file that user have filtered
      unordered_map<ULONGLONG, streampos> _filter_map;

      /// <summary>
      /// Read header from binary file
      /// </summary>
      void read_header();

      /// <summary>
      /// Read meta from binary file
      /// </summary>
      void read_meta();

      /// <summary>
      /// Fill internal buffer if we used all
      /// </summary>
      inline void fill_buffer();

      /// <summary>
      /// Make sure the file state correct.
      /// </summary>
      inline void ensure_file_opened();
    public:
      BinaryReader();
      ~BinaryReader();

      /// <summary>
      /// Open specified binary file.
      /// </summary>
      /// <param name="bin_file">Path to binary file.</param>
      void open(string bin_file);

      /// <summary>
      /// Close binary file, will disable furthur operations.
      /// </summary>
      void close();

      /// <summary>
      /// Get next item from binary file.
      ///
      /// NOTE:
      ///   This result container is share between each calling, so DO make sure copy the value, not this reference.
      /// </summary>
      /// <returns>Shared container to reader fields of item.</returns>
      ItemContainer* next_item();

      /// <summary>
      /// Get meta of the binary file.
      /// </summary>
      /// <returns>Meta from binary file.</returns>
      const Meta* get_meta();

      /// <summary>
      /// Get header of the binary file.
      /// </summary>
      /// <returns>Header from binary file.</returns>
      const BinHeader* get_header();

      /// <summary>
      /// Set filter for furthur reading.
      ///
      /// NOTE:
      ///   This function will affect the result of next_item.
      /// </summary>
      /// <param name="start">Start timestamp to filter.</param>
      /// <param name="end">End timestamp to filter.</param>
      void set_filter(ULONGLONG start, ULONGLONG end);

      /// <summary>
      /// Disable current filter.
      /// </summary>
      void disable_filter();

      /// <summary>
      /// Reset internal state.
      /// </summary>
      void reset();
    };
  } // namespace datalib

} // namespace maro

#endif
