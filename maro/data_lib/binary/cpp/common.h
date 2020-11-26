
#ifndef _MARO_DATALIB_COMMON_
#define _MARO_DATALIB_COMMON_

#include <unordered_map>
#include <string>
#include <ostream>

using namespace std;

namespace maro
{
    namespace datalib
    {
        using UCHAR = unsigned char;
        using ULONGLONG = unsigned long long;
        using LONGLONG = long long;
        using UINT = uint32_t;

        const string MARO = "maro";
        const unsigned char FILE_TYPE_BIN = 1;
        const unsigned char FILE_TYPE_INDEX = 2;
        const uint32_t CONVERTER_VERSION = 100;

        const int SECONDS_PER_HOUR = 60 * 60;

        const int BUFFER_LENGTH = 4096;

        const UCHAR DTYPE_CHAR = 1;
        const UCHAR DTYPE_UCHAR = 2;
        const UCHAR DTYPE_SHORT = 3;
        const UCHAR DTYPE_USHORT = 4;
        const UCHAR DTYPE_INT = 5;
        const UCHAR DTYPE_UINT = 6;
        const UCHAR DTYPE_LONG = 7;
        const UCHAR DTYPE_ULONG = 8;
        const UCHAR DTYPE_FLOAT = 9;
        const UCHAR DTYPE_DOUBLE = 10;
        const UCHAR DTYPE_TIME = 11;

        /*
        4 bytes - identifier "maro"
        1 byte - file type (0: reserved, 1: binary, 2: index)
        4 bytes - converter version
        4 bytes - file version
        2 bytes - custimized file type (2 char)
        8 bytes - total items
        4 bytes - item size
        1 byte - utc offset
        8 bytes - start timestamp (real)
        8 bytes - end timestamp (real)
        4 bytes - meta size (meta just follow header)
        32 bytes - reserved
        */
        struct BinHeader
        {
            unsigned char file_type;

            char custom_file_type[3]{"NA"};
            char identifier[5]{"MARO"};
            char utc_offset{0};

            UINT converter_version{0U};
            UINT file_version{0U};
            UINT item_size{0U};
            UINT meta_size{0ULL};

            ULONGLONG total_items{0ULL};
            ULONGLONG start_timestamp{0ULL};
            ULONGLONG end_timestamp{0ULL};

            ULONGLONG reserved1{0ULL};
            ULONGLONG reserved2{0ULL};
            ULONGLONG reserved3{0ULL};
            ULONGLONG reserved4{0ULL};

            friend ostream &operator<<(ostream &os, const BinHeader &header);
        };

        static unordered_map<string, pair<unsigned char, size_t>> field_dtype = {
            {"b", {DTYPE_CHAR, sizeof(char)}},
            {"B", {DTYPE_UCHAR, sizeof(unsigned char)}},
            {"s", {DTYPE_SHORT, sizeof(short)}},
            {"S", {DTYPE_USHORT, sizeof(unsigned short)}},
            {"i", {DTYPE_INT, sizeof(int32_t)}},
            {"I", {DTYPE_UINT, sizeof(uint32_t)}},
            {"l", {DTYPE_LONG, sizeof(LONGLONG)}},
            {"L", {DTYPE_ULONG, sizeof(ULONGLONG)}},
            {"f", {DTYPE_FLOAT, sizeof(float)}},
            {"d", {DTYPE_DOUBLE, sizeof(double)}},
            {"t", {DTYPE_TIME, sizeof(ULONGLONG)}},
        };

        struct Field
        {
            unsigned char type{2};
            uint32_t size{0U};
            uint32_t start_index{0U};
            string column;
            string alias;

            Field(string alias, string column, uint32_t size, uint32_t start_index, unsigned char dtype);
        };

        struct Meta
        {
            char utc_offset{0};

            vector<Field> fields;

            uint32_t itemsize();
        };

    } // namespace datalib

} // namespace maro

#endif