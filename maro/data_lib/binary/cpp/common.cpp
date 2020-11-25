
#include "common.h"

namespace maro
{
    namespace datalib
    {
        Field::Field(string alias, string column, uint32_t size, uint32_t start_index, unsigned char dtype)
            : alias(move(alias)),
              column(move(column)),
              size(move(size)),
              start_index(move(start_index)),
              type(move(dtype))
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

        ostream& operator<<(ostream& os, const BinHeader& header)
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
    } // namespace datalib

} // namespace maro
