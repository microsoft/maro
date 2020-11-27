// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "common.h"

namespace maro
{
  namespace datalib
  {
    Field::Field(string alias, string column, uint32_t size, uint32_t start_index, unsigned char dtype)
        : type(dtype),
          size(size),
          start_index(start_index),
          column(column),
          alias(alias)
    {
    }

    uint32_t Meta::itemsize()
    {
      uint32_t size = 0;

      for (auto f : fields)
      {
        size += f.size;
      }

      return size;
    }

    int Meta::size() const
    {
      return static_cast<int>(fields.size());
    }

    string Meta::get_alias(int field_index) const
    {
      auto &field = fields[field_index];

      return field.alias;
    }

    UCHAR Meta::get_type(int field_index) const
    {
      auto &field = fields[field_index];

      return field.type;
    }

    uint32_t Meta::get_start_index(int field_index) const
    {
      auto &field = fields[field_index];

      return field.start_index;
    }

    ostream &operator<<(ostream &os, const BinHeader &header)
    {
      os << "Identifier: " << header.identifier << endl;
      os << "File type: " << int(header.file_type) << endl;
      os << "Converter version: " << header.converter_version << endl;
      os << "File version: " << header.file_version << endl;
      os << "Customize file type: " << header.custom_file_type << endl;
      os << "Total items: " << header.total_items << endl;
      os << "Item size: " << header.item_size << endl;
      os << "UTC offset: " << int(header.utc_offset) << endl;
      os << "Start timestamp: " << header.start_timestamp << endl;
      os << "End timestamp: " << header.end_timestamp << endl;
      os << "Meta size: " << header.meta_size << endl;

      return os;
    }

    const char *InvalidTimestampDataType::what() const noexcept
    {
      return "Timestamp field must be t or L.";
    }

    const char *InvalidTimeToParse::what() const noexcept
    {
      return "Fail to parse datetime";
    }

    const char *ConvertVersionNotMatch::what() const noexcept
    {
      return "Binary converter version not match, please convert with same version.";
    }

    const char *FailToOpenBinaryFile::what() const noexcept
    {
      return "Fail to open binary file, please try again.";
    }

    const char *OperationBeforeFileOpen::what() const noexcept
    {
      return "Reading operations must after file opened.";
    }

    const char *BadBinaryFormat::what() const noexcept
    {
      return "Bad binary format to read header.";
    }

    const char *ConvertWithoutMeta::what() const noexcept
    {
      return "No meta file loaded.";
    }

    const char *FailToOpenCsvFile::what() const noexcept
    {
      return "Fail to open csv file.";
    }

    const char *MetaNoTimestamp::what() const noexcept
    {
      return "Meta must contains definition for timestamp.";
    }
  } // namespace datalib

} // namespace maro
