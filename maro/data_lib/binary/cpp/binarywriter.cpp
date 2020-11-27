// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "binarywriter.h"

namespace maro
{
  namespace datalib
  {
    // calculate utc offset for local timezone
    inline char calc_local_utc_offset()
    {
      time_t rawtime;
      struct tm* ptm;

      time(&rawtime);

      ptm = gmtime(&rawtime);

      auto t2 = mktime(ptm);

      return char((rawtime - t2) / SECONDS_PER_HOUR);
    }

    inline ULONGLONG to_timestamp(string& val_str, const string& format, unsigned char local_utc_offset = 0, char utc_offset = 0)
    {
      tm t{};
      istringstream ss(val_str);

      ss >> get_time(&t, format.c_str());

      // failed to parse time
      if (t.tm_year == 0 && t.tm_mon == 0 && t.tm_mday == 0)
      {
        throw InvalidTimeToParse();
      }

      auto t3 = mktime(&t);

      auto t4 = t3 + (long long(local_utc_offset) - utc_offset) * SECONDS_PER_HOUR;

      return ULONGLONG(t4);
    }

    inline char to_char(string& val_str)
    {
      return char(stoi(val_str));
    }

    inline unsigned char to_uchar(string& val_str)
    {
      return unsigned char(stoi(val_str));
    }

    inline short to_short(string& val_str)
    {
      return short(stoi(val_str));
    }

    inline unsigned short to_ushort(string& val_str)
    {
      return unsigned short(stoi(val_str));
    }

    inline int32_t to_int(string& val_str)
    {
      return int32_t(stoi(val_str));
    }

    inline uint32_t to_uint(string& val_str)
    {
      return uint32_t(stoul(val_str));
    }

    inline LONGLONG to_long(string& val_str)
    {
      return stoll(val_str);
    }

    inline ULONGLONG to_ulong(string& val_str)
    {
      return stoull(val_str);
    }

    inline float to_float(string& val_str)
    {
      return stof(val_str);
    }

    inline double to_double(string& val_str)
    {
      return stod(val_str);
    }

    BinaryWriter::BinaryWriter()
    {
      memset(_buffer, 0, BUFFER_LENGTH);
    }

    BinaryWriter::~BinaryWriter()
    {
      close();
    }

    void BinaryWriter::open(string output_file, string file_type, int32_t file_version)
    {
      local_utc_offset = calc_local_utc_offset();

      _file.open(output_file, ios::out | ios::binary);

      _is_opened = true;

      _header.file_type = FILE_TYPE_BIN;
      _header.converter_version = CONVERTER_VERSION;

      write_header();
    }

    void BinaryWriter::close()
    {
      if(_is_opened)
      {
        // update header before close
        write_header();

        _file.flush();
        _file.close();

        _is_opened = false;
      }
    }

    void BinaryWriter::load_meta(string meta_file)
    {
      ensure_file_opened();

      MetaParser parser;

      parser.parse(meta_file, _meta);

      _is_meta_loaded = true;

      _header.item_size = _meta.itemsize();
      _header.utc_offset = _meta.utc_offset;

      write_meta();

      write_header();
    }

    void BinaryWriter::add_csv(string csv_file)
    {
      ensure_file_opened();
      ensure_meta_loaded();

      CSV csv;

      if (csv.mmap(csv_file))
      {
        const auto& header = csv.header();

        // construct the column to field mapping for each csv file
        construct_column_mapping(header);

        // max number of items in buffer
        auto max_items_num = floorl(BUFFER_LENGTH / _header.item_size);

        // current items in buffer
        auto cur_items_num = 0;

        auto colum_index = 0;

        for (const auto& row : csv)
        {
          auto length = 0;

          auto is_valid_row = collect_item_to_buffer(row, cur_items_num);

          if (is_valid_row)
          {
            cur_items_num++;
            _header.total_items++;

            if (cur_items_num >= max_items_num)
            {
              _file.write(_buffer, ULONGLONG(cur_items_num) * _header.item_size);

              cur_items_num = 0;
            }

          }
        }

        if (cur_items_num != 0)
        {
          _file.write(_buffer, ULONGLONG(cur_items_num) * _header.item_size);
        }
      }
      else
      {
        throw FailToOpenCsvFile();
      }
    }

    void BinaryWriter::set_start_timestamp(ULONGLONG start_timestamp)
    {
      // NOTE: this function should be called before flush
      _header.start_timestamp = start_timestamp;
      _is_start_timestamp_set = true;
    }

    void BinaryWriter::construct_column_mapping(const CSV::Row& header)
    {
      // clear first
      _col2field_map.clear();

      auto hi = 0;

      // try to match the headers with meta, and keep the index
      for (const auto& h : header)
      {
        for (auto fi = 0; fi < _meta.fields.size(); fi++)
        {
          const auto& field = _meta.fields[fi];
          string hstr;

          h.read_value(hstr);

          if (hstr == field.column)
          {
            _col2field_map[hi] = fi;

            break;
          }
        }

        hi++;
      }
    }

#define WriteToBuffer(size, src)            \
    length = size;                          \
    memcpy(&_buffer[offset], &src, length); \
    offset += length;

    void BinaryWriter::write_header()
    {
      _file.seekp(0, ios::beg);

      size_t offset = 0ULL;
      size_t length = 0ULL;

      WriteToBuffer(strlen(_header.identifier), _header.identifier)
      WriteToBuffer(sizeof(unsigned char), _header.file_type)
      WriteToBuffer(sizeof(UINT), _header.converter_version)
      WriteToBuffer(sizeof(UINT), _header.file_version)
      WriteToBuffer(strlen(_header.custom_file_type), _header.custom_file_type)
      WriteToBuffer(sizeof(ULONGLONG), _header.total_items)
      WriteToBuffer(sizeof(UINT), _header.item_size)
      WriteToBuffer(sizeof(char), _header.utc_offset)
      WriteToBuffer(sizeof(ULONGLONG), _header.start_timestamp)
      WriteToBuffer(sizeof(ULONGLONG), _header.end_timestamp)
      WriteToBuffer(sizeof(ULONGLONG), _header.meta_size)

      WriteToBuffer(sizeof(ULONGLONG), _header.reserved1)
      WriteToBuffer(sizeof(ULONGLONG), _header.reserved2)
      WriteToBuffer(sizeof(ULONGLONG), _header.reserved3)
      WriteToBuffer(sizeof(ULONGLONG), _header.reserved4)

      _file.write(_buffer, offset);
      _file.seekp(0, ios::end);
    }

    void BinaryWriter::write_meta()
    {
      /*
      Each field info if a pair of name and index:

      4 bytes - start index (in bytes) in binary block
      1 byte - filed type
      2 bytes - field name length
      N bytes - filed name that length same as above specified
      */

      for (auto& field : _meta.fields)
      {
        size_t offset = 0ULL;
        size_t length = 0ULL;

        WriteToBuffer(sizeof(uint32_t), field.start_index)
        WriteToBuffer(sizeof(unsigned char), field.type)
        WriteToBuffer(sizeof(uint32_t), field.size)

        auto alias_length = field.alias.size();

        WriteToBuffer(sizeof(unsigned short), alias_length)
        WriteToBuffer(alias_length, field.alias[0])

        _file.write(_buffer, offset);

        _header.meta_size += offset;
      }
    }

#define WriteField(to_type_func, dtype)             \
    auto rv = to_type_func(v);                      \
    memcpy(&_buffer[offset], &rv, sizeof(dtype));   \
    is_valid_row = true;                            

    inline bool BinaryWriter::collect_item_to_buffer(CSV::Row row, int cur_items_num)
    {
      auto column_index = 0;

      auto is_valid_row = false;

      for (const auto& cell : row)
      {
        auto iter = _col2field_map.find(column_index);

        if (iter != _col2field_map.end())
        {
          string v;

          // we find the column
          cell.read_value(v);

          auto& field = _meta.fields[iter->second];
          auto offset = cur_items_num * _header.item_size + field.start_index;

          // timestamp must be L or t
          if (field.alias == "timestamp" && (field.type != DTYPE_ULONG && field.type != DTYPE_TIME))
          {
            throw InvalidTimestampDataType();
          }

          switch (field.type)
          {
          case DTYPE_CHAR: { WriteField(to_char, char) break; }
          case DTYPE_UCHAR: { WriteField(to_uchar, unsigned char) break; }
          case DTYPE_SHORT: { WriteField(to_short, short) break; }
          case DTYPE_USHORT: { WriteField(to_ushort, short) break; }
          case DTYPE_INT: { WriteField(to_int, int32_t) break; }
          case DTYPE_UINT: { WriteField(to_uint, uint32_t) break; }
          case DTYPE_LONG: { WriteField(to_long, LONGLONG) break; }
          case DTYPE_FLOAT: { WriteField(to_float, float) break; }
          case DTYPE_DOUBLE: { WriteField(to_double, double) break; }
          case DTYPE_ULONG:
          case DTYPE_TIME:
          {
            ULONGLONG rv = 0ULL;

            if (field.type == DTYPE_ULONG)
            {
              rv = to_ulong(v);
            }
            else {
              rv = convert_to_timestamp(v);
            }

            memcpy(&_buffer[offset], &rv, sizeof(ULONGLONG));

            // update header
            if (field.alias == "timestamp")
            {
              if (!_is_start_timestamp_set)
              {
                _header.start_timestamp = rv;

                _is_start_timestamp_set = true;
              }

              _header.end_timestamp = rv;
            }

            break;
          }
          default:
            break;
          }
        }

        column_index++;
      }

      return is_valid_row;
    }

    inline void BinaryWriter::ensure_file_opened()
    {
      if (!_is_opened)
      {
        throw OperationBeforeFileOpen();
      }
    }

    inline void BinaryWriter::ensure_meta_loaded()
    {
      if (!_is_meta_loaded)
      {
        throw ConvertWithoutMeta();
      }
    }

    inline ULONGLONG BinaryWriter::convert_to_timestamp(string& val_str)
    {
      string format = _meta.format == "" ? DEFAULT_FORMAT : _meta.format;

      return to_timestamp(val_str, format, local_utc_offset, _meta.utc_offset);
    }

  } // namespace datalib
} // namespace maro
