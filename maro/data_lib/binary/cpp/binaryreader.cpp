

#include "binaryreader.h"

namespace maro
{
    namespace datalib
    {
        BinaryReaderIterator::BinaryReaderIterator()
        {
        }

        BinaryReaderIterator::~BinaryReaderIterator()
        {
            _reader = nullptr;
        }

        void BinaryReaderIterator::set_reader(BinaryReader *reader)
        {
            _reader = reader;
        }

        ItemContainer *BinaryReaderIterator::operator*()
        {
            return _reader->next_item();
        }

        BinaryReaderIterator &BinaryReaderIterator::operator++()
        {
            return *this;
        }

        bool BinaryReaderIterator::operator!=(const BinaryReaderIterator &bri)
        {
            return !_reader->_file.eof() || (_reader->cur_item_index >= 0 && _reader->cur_item_index < _reader->max_items_in_buffer);
        }

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

            cur_item_index++;

            return &_item;
        }

        const Meta *BinaryReader::get_meta()
        {
            return &_meta;
        }

        BinaryReaderIterator BinaryReader::begin()
        {
            BinaryReaderIterator iter;

            iter.set_reader(this);

            return iter;
        }

        BinaryReaderIterator BinaryReader::end()
        {
            BinaryReaderIterator iter;

            iter.set_reader(this);

            return iter;
        }

        void BinaryReader::reset()
        {
            _file.clear();
            _file.seekg(_data_offset, ios::beg);

            cur_item_index = -1;

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
