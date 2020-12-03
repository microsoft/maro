#ifndef _MARO_DATALIB_BINARY_COMBINE_
#define _MARO_DATALIB_BINARY_COMBINE_

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
    /// Util used to combine 2 binary horizontally.
    /// </summary>
    class BinaryCombine
    {
    public:
      BinaryCombine();
      ~BinaryCombine();

      /// <summary>
      /// Open output file for writing.
      /// </summary>
      /// <param name="output_file">Path to output file.</param>
      void open(string output_file);

      /// <summary>
      /// Close output file.
      /// </summary>
      void close();

      /// <summary>
      /// Combine 2 input binary file into output file.
      /// </summary>
      /// <param name="bin_file1">First binary file to combine, output file will use header and meta of this file.</param>
      /// <param name="bin_file2">Seconds binary file to combine.</param>
      void combine(string bin_file1, string bin_file2);

    private:
      ofstream _file;

      // Is output file opened.
      bool _is_opened = false;

      // Current total items.
      ULONGLONG _total_items = 0ULL;

      // Min start timestamp for 2 binary files.
      ULONGLONG _start_timestamp = 0ULL;

      // Max end timestamp for 2 binary file.
      ULONGLONG _end_timestamp = 0ULL;

      // Buffer used to read from 1st binary file.
      char _buffer1[BUFFER_LENGTH];

      // Buffer used to read from 2nd binary file.
      char _buffer2[BUFFER_LENGTH];
    };


  }
}


#endif // !_MARO_DATALIB_BINARY_COMBINE_
