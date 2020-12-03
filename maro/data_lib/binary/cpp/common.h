// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

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
    using USHORT = unsigned short;
    using ULONGLONG = unsigned long long;
    using LONGLONG = long long;
    using UINT = uint32_t;

    const string MARO = "maro";
    const unsigned char FILE_TYPE_BIN = 1;
    const unsigned char FILE_TYPE_INDEX = 2;
    const uint32_t CONVERTER_VERSION = 100;

    const int SECONDS_PER_HOUR = 60 * 60;

    const UINT BUFFER_LENGTH = 4096;

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

    // NOTE: this must be update if header has changes
    const UCHAR HEADER_LENGTH = 84;
    const UCHAR HEADER_TOTAL_ITEMS_OFFSET = 15;
    const UCHAR HEADER_ITEM_SIZE_OFFSET = 23;
    const UCHAR HEADER_START_TIME_OFFSET = 28;
    const UCHAR HEADER_END_TIME_OFFSET = 36;
    const UCHAR HEADER_META_SIZE_OFFSET = 44;

    /*
    Header in binary file, contains following items:

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
      UCHAR file_type = 0;

      char custom_file_type[3] = "NA";
      char identifier[5] = "MARO";
      char utc_offset = 0;

      UINT converter_version = 0U;
      UINT file_version = 0U;
      UINT item_size = 0U;
      UINT meta_size = 0ULL;

      ULONGLONG total_items = 0ULL;
      ULONGLONG start_timestamp = 0ULL;
      ULONGLONG end_timestamp = 0ULL;

      ULONGLONG reserved1 = 0ULL;
      ULONGLONG reserved2 = 0ULL;
      ULONGLONG reserved3 = 0ULL;
      ULONGLONG reserved4 = 0ULL;

      friend ostream& operator<<(ostream& os, const BinHeader& header);
    };

    // data type definition we supported
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

    // Field definition from meta
    struct Field
    {
      UCHAR type = 0;
      uint32_t size = 0U;
      uint32_t start_index = 0U;
      string column;
      string alias;

      Field(string alias, string column, uint32_t size, uint32_t start_index, unsigned char dtype);
    };

    // Meta from meta.toml
    struct Meta
    {
      char utc_offset = 0;

      string format;

      vector<Field> fields;

      uint32_t itemsize();

      int size() const;

      string get_alias(int field_index) const;

      UCHAR get_type(int field_index) const;

      uint32_t get_start_index(int field_index) const;
    };

    // Data type of timestamp not correct.
    class InvalidTimestampDataType : public exception
    {
    public:
      const char* what() const noexcept override;
    };

    // Fail to parse the datetime string
    class InvalidTimeToParse : public exception
    {
    public:
      const char* what() const noexcept override;
    };

    // Converter vesion not match current.
    class ConvertVersionNotMatch : public exception
    {
    public:
      const char* what() const noexcept override;
    };

    // Fail to open binary file, may be not exist
    class FailToOpenBinaryFile : public exception
    {
    public:
      const char* what() const noexcept override;
    };

    // File not opened, but do operations
    class OperationBeforeFileOpen : public exception
    {
    public:
      const char* what() const noexcept override;
    };

    // Binary format of the binary file not correct.
    class BadBinaryFormat : public exception
    {
    public:
      const char* what() const noexcept override;
    };

    // No meta to convert.
    class ConvertWithoutMeta : public exception
    {
    public:
      const char* what() const noexcept override;
    };

    // Fail to open the csv file.
    class FailToOpenCsvFile : public exception
    {
    public:
      const char* what() const noexcept override;
    };

    // No timestamp definition in meta
    class MetaNoTimestamp : public exception
    {
    public:
      const char* what() const noexcept override;
    };
  } // namespace datalib

} // namespace maro

#endif
