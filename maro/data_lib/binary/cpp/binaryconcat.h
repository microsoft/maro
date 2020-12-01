#ifndef _MARO_DATALIB_BINARY_CONCAT_
#define _MARO_DATALIB_BINARY_CONCAT_

#include <string>
#include <iostream>
#include <fstream>

#include "common.h"


using namespace std;


namespace maro
{
  namespace datalib
  {
    /// <summary>
    /// Util used to concat binary files one after another (vertically)
    /// </summary>
    class BinaryConcat
    {
    public:
      BinaryConcat();

      void open(string out_file);
      void add(string bin_file);
      void close();
    private:
      ofstream _file;
      char _buffer[BUFFER_LENGTH];

      // Is current file the first one, we use its header as output file
      bool _is_first_file = true;

      bool _is_opened = false;

      ULONGLONG _total_items = 0ULL;
      UINT  _meta_size = 0U;

      inline void ensure_file_opened();
    };

  }
}

#endif
