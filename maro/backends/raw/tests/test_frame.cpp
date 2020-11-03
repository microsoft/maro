#include "lest.hpp"
#include "../frame.h"


using namespace std;
using namespace maro::backends::raw;

const lest::test specification[] =
{
  CASE("")
  {

  },
};



int main(int argc, char* argv[])
{
  return lest::run(specification, argc, argv);
}
