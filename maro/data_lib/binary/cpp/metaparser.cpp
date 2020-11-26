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

                auto kv = field_dtype.find(type);
                auto size = uint32_t(kv->second.second);

                if(alias == "timestamp")
                {
                    has_timestamp = true;

                    // check the data type


                    // make sure timestamp is the first field write to binary file
                    meta.fields.emplace(meta.fields.begin(), alias, col_name, size, offset, kv->second.first);
                }
                else
                {
                    // push to back
                    meta.fields.emplace_back(alias, col_name, size, offset, kv->second.first);
                }
                

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
                std::cerr << "Cannot find UTC offset, use 0." << endl;

                meta.utc_offset = 0;
            }
            
            // try to get format
            try
            {
                meta.format = toml::find<string>(data, "format");
            }
            catch(std::out_of_range)
            {
                std::cerr << "Cannot find datetime format, use default." << endl;
            }
        }

    } // namespace datalib

} // namespace maro
