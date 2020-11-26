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

      void read_header();
      void read_meta();

      inline void fill_buffer();

      inline void ensure_file_opened();
    public:
      BinaryReader();
      ~BinaryReader();

      void open(string bin_file);

      ItemContainer* next_item();

      const Meta* get_meta();
      const BinHeader* get_header();

      void set_filter(ULONGLONG start, ULONGLONG end);
      void disable_filter();

      void reset();
    };
  } // namespace datalib

} // namespace maro

#endif
