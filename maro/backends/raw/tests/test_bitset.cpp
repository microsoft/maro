
#include <iostream>

#include "../bitset.h"
#include "lest.hpp"

using namespace std;
using namespace maro::backends::raw;


const lest::test specification[] = 
{
    CASE("Bitset initial mask size (bit) should be times of 64.")
    {
        auto bs = Bitset(10);

        EXPECT(1 == bs.mask_size());
        EXPECT(64 == bs.size());

        auto bs2 = Bitset(128);

        EXPECT(2 == bs.mask_size());
        EXPECT(128 == bs.size());
    }
};


int main( int argc, char * argv[] )
{
    return lest::run( specification, argc, argv );
}