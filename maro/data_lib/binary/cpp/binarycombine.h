#ifndef _MARO_DATALIB_BINARY_COMBINE_
#define _MARO_DATALIB_BINARY_COMBINE_

#include <string>
#include <iostream>
#include <fstream>

#include "common.h"

namespace maro
{
  namespace datalib
  {

    class BinaryCombine
    {
    public:
      BinaryCombine();

      void open();

      void close();

      void add();
    };
  }
}


#endif // !_MARO_DATALIB_BINARY_COMBINE_
