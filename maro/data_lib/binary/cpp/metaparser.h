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
    class MetaParser
    {

    public:
      MetaParser();

      /// <summary>
      /// Parse toml file to retrieve meta.
      /// </summary>
      /// <param name="meta_file">Path to the toml file.</param>
      /// <param name="meta">Meta object for result.</param>
      void parse(string meta_file, Meta& meta);
    };
  } // namespace datalib

} // namespace maro

#endif
