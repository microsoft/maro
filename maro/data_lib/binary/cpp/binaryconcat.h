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
      ~BinaryConcat();

      /// <summary>
      /// Open output file for writing.
      /// </summary>
      /// <param name="out_file">Path to output file.</param>
      void open(string out_file);

      /// <summary>
      /// Append binary file to the tail.
      /// </summary>
      /// <param name="bin_file">Path to binary file to append.</param>
      void add(string bin_file);

      /// <summary>
      /// Close output file.
      /// </summary>
      void close();
    private:
      ofstream _file;

      // Reading buffer
      char _buffer[BUFFER_LENGTH];

      // Is current file the first one, we use its header as output file
      bool _is_first_file = true;

      // Is file opened
      bool _is_opened = false;

      // Total items in output file
      ULONGLONG _total_items = 0ULL;

      // size of meta part
      UINT  _meta_size = 0U;

      /// <summary>
      /// Ensure that the output file is opened
      /// </summary>
      inline void ensure_file_opened();
    };

  }
}

#endif
