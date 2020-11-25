#include "metaparser.h"

namespace maro
{
    namespace datalib
    {
        MetaParser::MetaParser()
        {
        }

        void MetaParser::parse(string meta_file, Meta &meta)
        {
            // load the file
            const auto data = toml::parse(meta_file);

            // find row definition
            const auto row = toml::find<vector<toml::table>>(data, "row");

            uint32_t offset = 0U;

            bool has_timestamp = false;

            for (auto col : row)
            {
                // parse each column definition
                auto col_name = col["column"].as_string();
                auto type = col["type"].as_string();
                auto alias = col["alias"].as_string();

                if(alias == "timestamp")
                {
                    has_timestamp = true;

                    // check the data type
                    if(type != "t")
                    {
                        throw out_of_range("Incorrect timestamp type.");
                    }
                }

                auto kv = field_dtype.find(type);
                auto size = uint32_t(kv->second.second);

                meta.fields.emplace_back(alias, col_name, size, offset, kv->second.first);

                offset += size;
            }

            if(!has_timestamp)
            {
                throw overflow_error("Must contains timestamp definition.");
            }

            try
            {
                meta.utc_offset = toml::find<char>(data, "utc_offset");
            }
            catch(std::out_of_range)
            {
                std::cerr << "Cannot find UTC offset, use 0." << '\n';
            }
            
            
        }

    } // namespace datalib

} // namespace maro
