// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.


#include "binaryreader.h"

namespace maro
{
  namespace datalib
  {
    BinaryReader::BinaryReader()
    {
      memset(_buffer, 0, BUFFER_LENGTH);
    }

    void BinaryReader::open(string bin_file)
    {
      _file.open(bin_file, ios::binary | ios::in);

      // check if file opened correct
      if (!_file.is_open())
      {
        throw FailToOpenBinaryFile();
      }

      _is_opened = true;

      read_header();
      read_meta();

      _data_offset = _file.tellg();

      max_items_in_buffer = floorl(BUFFER_LENGTH / _header.item_size);

      _item.set_buffer(_buffer);
    }

    BinaryReader::~BinaryReader()
    {
      _file.close();
    }

    ItemContainer* BinaryReader::next_item()
    {
      ensure_file_opened();

      fill_buffer();

      if (cur_item_index < 0)
      {
        return nullptr;
      }

      // set offset to pointer to current item
      _item.set_offset(cur_item_index * _header.item_size);

      // check if current timestamp less than filter end if specified
      if (_is_filter_enabled && _filter_end != INVALID_FILTER && _filter_end > _filter_start)
      {
        // check if reach the end
        // NOTE: first field always be timestamp
        auto cur_end = _item.get<ULONGLONG>(0);

        if (cur_end >= _filter_end)
        {
          return nullptr;
        }
      }

      cur_item_index++;

      return &_item;
    }

    const Meta* BinaryReader::get_meta()
    {
      ensure_file_opened();

      return &_meta;
    }

    const BinHeader* BinaryReader::get_header()
    {
      ensure_file_opened();

      return &_header;
    }

    void BinaryReader::set_filter(ULONGLONG start, ULONGLONG end)
    {
      ensure_file_opened();

      _is_filter_enabled = true;

      _filter_start = start;
      _filter_end = end;

      // check if we have this filter before?
      auto iter = _filter_map.find(start);

      if (iter != _filter_map.end())
      {
        // seek to if we have this filter before
        _file.seekg(iter->second);

        cur_item_index = -1;
      }
      else
      {
        // try to find it
        ItemContainer* item = next_item();

        auto i = 0ULL;

        while (item != nullptr)
        {
          i++;

          // first 8 bytes if timestamp for each item
          if (item->get<ULONGLONG>(0) >= start)
          {
            // move back for furthur operation
            _file.seekg(ULONGLONG(_data_offset) + _header.item_size * (i - 1));

            _filter_map[start] = _file.tellg();

            // force re-fill buffer
            cur_item_index = -1;

            break;
          }

          item = next_item();
        }
      }
    }

    void BinaryReader::disable_filter()
    {
      _is_filter_enabled = false;
    }

    void BinaryReader::reset()
    {
      ensure_file_opened();

      // NOTE: stream must be cleared, or we cannot get correct gcout value
      _file.clear();
      _file.seekg(_data_offset, ios::beg);

      cur_item_index = -1;

      _is_filter_enabled = false;

      _filter_start = INVALID_FILTER;
      _filter_end = INVALID_FILTER;

      max_items_in_buffer = floorl(BUFFER_LENGTH / _header.item_size);
    }

    inline void BinaryReader::fill_buffer()
    {
      if (cur_item_index < 0 || cur_item_index >= max_items_in_buffer)
      {
        if (_file.eof())
        {
          cur_item_index = -1;
        }
        else
        {
          // read into buffer
          _file.read(_buffer, ULONGLONG(max_items_in_buffer) * _header.item_size);

          // update max items in buffer according to bytes we readed
          max_items_in_buffer = min<long>(max_items_in_buffer, floor(_file.gcount() / _header.item_size));

          cur_item_index = max_items_in_buffer == 0 ? -1 : 0;
        }
      }
    }

    inline void BinaryReader::ensure_file_opened()
    {
      if (!_is_opened)
      {
        throw OperationBeforeFileOpen();
      }
    }

#define ReaderHeaderNormalItem(field, type)         \
  length = sizeof(type);                            \
  memcpy(&_header.field, &buffer[offset], length);  \
  offset += length;

#define ReaderHeaderStringItem(field)               \
  length = strlen(_header.field);                   \
  memcpy(_header.field, &buffer[offset], length);   \
  offset += length;

    void BinaryReader::read_header()
    {
      char buffer[HEADER_LENGTH];

      if (_file.read(buffer, HEADER_LENGTH))
      {
        auto offset = 0;
        auto length = 0;

        ReaderHeaderStringItem(identifier)
        ReaderHeaderNormalItem(file_type, UCHAR)
        ReaderHeaderNormalItem(converter_version, UINT)
        ReaderHeaderNormalItem(file_version, UINT)
        ReaderHeaderStringItem(custom_file_type)

        ReaderHeaderNormalItem(total_items, ULONGLONG)
        ReaderHeaderNormalItem(item_size, UINT)

        ReaderHeaderNormalItem(utc_offset, char)

        ReaderHeaderNormalItem(start_timestamp, ULONGLONG)
        ReaderHeaderNormalItem(end_timestamp, ULONGLONG)

        ReaderHeaderNormalItem(meta_size, ULONGLONG)

        ReaderHeaderNormalItem(reserved1, ULONGLONG)
        ReaderHeaderNormalItem(reserved2, ULONGLONG)
        ReaderHeaderNormalItem(reserved3, ULONGLONG)
        ReaderHeaderNormalItem(reserved4, ULONGLONG)
      }
      else
      {
        throw BadBinaryFormat();
      }

      if (_header.converter_version != CONVERTER_VERSION)
      {
        throw ConvertVersionNotMatch();
      }
    }

#define ReadMetaNormalItem(field, type)       \
  length = sizeof(type);                      \
  memcpy(&field, &buffer[offset], length);    \
  offset += length;

    void BinaryReader::read_meta()
    {
      // meta binary buffer
      unique_ptr<char> buffer_ptr = make_unique<char>(_header.meta_size);
      char* buffer = buffer_ptr.get();

      if (_file.read(buffer, _header.meta_size))
      {
        // meta fields
        uint32_t start_index = 0;
        unsigned char type = 0;
        unsigned short alias_length = 0;
        uint32_t size = 0;
        string alias;

        // offset read from buffer
        UINT offset = 0;
        UINT length = 0;

        while (offset < _header.meta_size)
        {
          ReadMetaNormalItem(start_index, uint32_t)
          ReadMetaNormalItem(type, UCHAR)
          ReadMetaNormalItem(size, uint32_t)
          ReadMetaNormalItem(alias_length, unsigned short)

          alias.resize(alias_length);

          memset(&alias[0], 0, alias_length);
          memcpy(&alias[0], &buffer[offset], alias_length);

          offset += alias_length;

          _meta.fields.emplace_back(alias, "", size, start_index, type);
        }
      }
      else
      {
        throw BadBinaryFormat();
      }
    }

  } // namespace datalib

} // namespace maro
