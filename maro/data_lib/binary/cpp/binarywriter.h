// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef _MARO_DATALIB_BINARY_WRITER_
#define _MARO_DATALIB_BINARY_WRITER_

#include <time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <map>

#include "csv2/csv2.hpp"
#include "metaparser.h"

using namespace std;

using CSV = csv2::Reader<csv2::delimiter<','>,
  csv2::quote_character<'\"'>,
  csv2::first_row_is_header<true>,
  csv2::trim_policy::trim_characters<' ', '\"', '\r'>>;

namespace maro
{
  namespace datalib
  {
    const string DEFAULT_FORMAT = "%Y-%m-%d %H:%M:%S";

    class BinaryWriter
    {
    public:
      BinaryWriter();
      BinaryWriter(const BinaryWriter& writer) = delete;

      ~BinaryWriter();

      /// <summary>
      /// Open a file to hold result binary.
      /// </summary>
      /// <param name="output_file">Path to the output binary file.</param>
      /// <param name="file_type">Customized file type.</param>
      /// <param name="file_version">Version of the file.</param>
      void open(string output_file, string file_type = "NA", int32_t file_version = 0);

      /// <summary>
      /// Close the file.
      /// </summary>
      void close();

      /// <summary>
      /// Load meta to prepare converting.
      /// </summary>
      /// <param name="meta_file">Path to meta file.</param>
      void load_meta(string meta_file);

      /// <summary>
      /// Load specified csv file, and convert it into binary.
      /// </summary>
      /// <param name="csv_file">Path to csv file to convert.</param>
      void add_csv(string csv_file);

      /// <summary>
      /// Customize the start timestamp in binary file, default is the timestamp of first item.
      /// </summary>
      /// <param name="start_timestamp">Timestamp to set.</param>
      void set_start_timestamp(ULONGLONG start_timestamp);

    private:
      // utc offset for local timezone
      char local_utc_offset = CHAR_MIN;

      // if we have set start timestamp before
      bool _is_start_timestamp_set = false;

      // seems FILE is faster than ofstream
      ofstream _file;

      // header to write
      BinHeader _header;

      // if output binary file opened
      bool _is_opened{ false };

      // if meta file loaded
      bool _is_meta_loaded{ false };

      // meta to write
      Meta _meta;

      // internal buffer for writing
      char _buffer[BUFFER_LENGTH];

      // used to map column index to field index
      map<int, int> _col2field_map;

      /// <summary>
      /// Create mapping from column index to field index.
      /// </summary>
      void construct_column_mapping(const CSV::Row& header);

      /// <summary>
      /// Write header to output file.
      /// </summary>
      void write_header();

      /// <summary>
      /// Write meta to output file.
      /// </summary>
      void write_meta();

      /// <summary>
      /// Convert a string into timestamp.
      /// </summary>
      /// <param name="val_str">String to convert.</param>
      /// <returns>Timestamp in UCT.</returns>
      inline ULONGLONG convert_to_timestamp(string& val_str);

      /// <summary>
      /// Collect a row from csv into internal buffer.
      /// </summary>
      /// <param name="row">Row to parse.</param>
      /// <param name="cur_items_num">How many items we wrote into buffer.</param>
      /// <returns>Is this row is valid.</returns>
      inline bool collect_item_to_buffer(CSV::Row row, int cur_items_num);

      /// <summary>
      /// Make sure the file state if correct.
      /// </summary>
      inline void ensure_file_opened();

      /// <summary>
      /// Make sure the meta is loaded.
      /// </summary>
      inline void ensure_meta_loaded();
    };
  } // namespace datalib

} // namespace maro

#endif
