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

        class ConvertVersionNotMatch : public exception
        {
        };

        class BinaryReader
        {
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

            bool _is_filter_enabled{false};
            ULONGLONG _filter_start{INVALID_FILTER};
            ULONGLONG _filter_end{INVALID_FILTER};

            // used to save the offset in file that user have filtered
            unordered_map<ULONGLONG, streampos> _filter_map;

            void read_header();
            void read_meta();

            void fill_buffer();

        public:
            BinaryReader();
            ~BinaryReader();

            void open(string bin_file);

            ItemContainer *next_item();

            const Meta *get_meta();

            void set_filter(ULONGLONG start, ULONGLONG end);
            void disable_filter();

            void reset();
        };
    } // namespace datalib

} // namespace maro

#endif