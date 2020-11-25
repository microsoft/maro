// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef _MARO_DATALIB_METAPARSER_
#define _MARO_DATALIB_METAPARSER_

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>

#include "toml11/toml.hpp"
#include "common.h"

using namespace std;

namespace maro
{
    namespace datalib
    {
        // load meta and parse, return fields info
        class MetaParser
        {

        public:
            MetaParser();

            void parse(string meta_file, Meta &meta);
        };
    } // namespace datalib

} // namespace maro

#endif