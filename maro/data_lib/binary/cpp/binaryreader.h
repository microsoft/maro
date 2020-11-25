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

        class ConvertVersionNotMatch : public exception
        {
        };

        class BinaryReaderIterator;

        class BinaryReader
        {
            friend BinaryReaderIterator;

        private:
            BinHeader _header;
            Meta _meta;

            ifstream _file;

            char _buffer[BUFFER_LENGTH];
            ItemContainer _item;

            long max_items_in_buffer{0};
            int cur_item_index{-1};

            // offset of data part
            streampos _data_offset{0};

            // used to save the offset in file that user have filtered
            unordered_map<streamoff, streamoff> _filter_map;

            void read_header();
            void read_meta();

            void fill_buffer();

        public:
            BinaryReader(string bin_file);
            ~BinaryReader();

            ItemContainer *next_item();

            const Meta *get_meta();

            BinaryReaderIterator begin();

            BinaryReaderIterator end();

            void reset();
        };

        class BinaryReaderIterator
        {
            BinaryReader *_reader;

        public:
            BinaryReaderIterator(BinaryReader *_reader);
            ~BinaryReaderIterator();

            // move to next
            BinaryReaderIterator &operator++();

            // get current item
            ItemContainer *operator*();

            bool operator!=(const BinaryReaderIterator &bri);
        };

    } // namespace datalib

} // namespace maro

#endif