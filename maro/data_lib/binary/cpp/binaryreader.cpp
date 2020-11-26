// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.


#include "binaryreader.h"

namespace maro
{
    namespace datalib
    {
        BinaryReader::BinaryReader()
        {
        }

        void BinaryReader::open(string bin_file)
        {
            _file.open(bin_file, ios::binary | ios::in);

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

        ItemContainer *BinaryReader::next_item()
        {
            fill_buffer();

            if (cur_item_index < 0)
            {
                return nullptr;
            }

            _item.set_offset(cur_item_index * _header.item_size);

            if (_is_filter_enabled && _filter_end != INVALID_FILTER && _filter_end > _filter_start)
            {
                // check if reach the end
                auto cur_end = _item.get<ULONGLONG>(0);

                if (cur_end >= _filter_end)
                {
                    return nullptr;
                }
            }

            cur_item_index++;

            return &_item;
        }

        const Meta *BinaryReader::get_meta()
        {
            return &_meta;
        }

        const BinHeader *BinaryReader::get_header()
        {
            return &_header;
        }

        void BinaryReader::set_filter(ULONGLONG start, ULONGLONG end)
        {
            _is_filter_enabled = true;

            _filter_start = start;
            _filter_end = end;

            // check if we have this filter before?
            auto iter = _filter_map.find(start);

            if (iter != _filter_map.end())
            {
                // seek to
                _file.seekg(iter->second);

                cur_item_index = -1;
            }
            else
            {
                // try to find it
                ItemContainer *item = next_item();

                auto i = 0;

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
            _file.clear();
            _file.seekg(_data_offset, ios::beg);

            cur_item_index = -1;

            _is_filter_enabled = false;

            _filter_start = INVALID_FILTER;
            _filter_end = INVALID_FILTER;

            max_items_in_buffer = floorl(BUFFER_LENGTH / _header.item_size);
        }

        void BinaryReader::fill_buffer()
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
                    _file.read(_buffer, max_items_in_buffer * _header.item_size);

                    max_items_in_buffer = min<long>(max_items_in_buffer, floor(_file.gcount() / _header.item_size));

                    cur_item_index = max_items_in_buffer == 0 ? -1 : 0;
                }
            }
        }

        void BinaryReader::read_header()
        {
            _file.read(_header.identifier, strlen(_header.identifier));
            _file.read((char *)(&_header.file_type), sizeof(unsigned char));
            _file.read((char *)(&_header.converter_version), sizeof(UINT));
            _file.read((char *)(&_header.file_version), sizeof(UINT));
            _file.read(_header.custom_file_type, strlen(_header.custom_file_type));
            _file.read((char *)(&_header.total_items), sizeof(ULONGLONG));
            _file.read((char *)(&_header.item_size), sizeof(UINT));
            _file.read(&_header.utc_offset, sizeof(char));
            _file.read((char *)(&_header.start_timestamp), sizeof(ULONGLONG));
            _file.read((char *)(&_header.end_timestamp), sizeof(ULONGLONG));
            _file.read((char *)(&_header.meta_size), sizeof(ULONGLONG));
            _file.read((char *)(&_header.reserved1), sizeof(ULONGLONG));
            _file.read((char *)(&_header.reserved2), sizeof(ULONGLONG));
            _file.read((char *)(&_header.reserved3), sizeof(ULONGLONG));
            _file.read((char *)(&_header.reserved4), sizeof(ULONGLONG));

            if (_header.converter_version != CONVERTER_VERSION)
            {
                throw ConvertVersionNotMatch();
            }
        }

        void BinaryReader::read_meta()
        {
            UINT read_size = 0;

            while (read_size < _header.meta_size)
            {
                size_t length = 0;

                uint32_t start_index = 0;
                unsigned char type = 0;
                unsigned short alias_length = 0;
                uint32_t size = 0;
                string alias;

                length = sizeof(uint32_t);
                _file.read((char *)&start_index, length);
                read_size += length;

                length = sizeof(unsigned char);
                _file.read((char *)&type, length);
                read_size += length;

                length = sizeof(uint32_t);
                _file.read((char *)&size, sizeof(uint32_t));
                read_size += length;

                length = sizeof(unsigned short);
                _file.read((char *)&alias_length, sizeof(unsigned short));
                read_size += length;

                alias.resize(alias_length);

                _file.read(&alias[0], alias_length);

                read_size += alias_length;

                _meta.fields.emplace_back(alias, "", size, start_index, type);
            }
        }

    } // namespace datalib

} // namespace maro
