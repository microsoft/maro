// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <iostream>

#include "../bitset.h"
#include "lest.hpp"

using namespace std;
using namespace maro::backends::raw;


const lest::test specification[] =
{
    CASE("Initial size should same as passed in if its times of 64.")
    {
        auto bs = Bitset(128);

        EXPECT(2 == bs.mask_size());
        EXPECT(128 == bs.size());
    },

    CASE("Initial size should be corrected to nearest (ceiling) times of 64.")
    {
        auto bs = Bitset(10);

        EXPECT(1 == bs.mask_size());
        EXPECT(64 == bs.size());
    },

    CASE("Initial size equals 0 is correct.")
    {
        auto bs = Bitset(0);

        EXPECT(0 == bs.mask_size());
        EXPECT(0 == bs.size());
    },

    CASE("Default value should be false.")
    {
        auto bs = Bitset(1);

        for (auto i = 0; i < bs.size(); i++)
        {
            EXPECT(false == bs.get(i));
        }

    },

    CASE("Invalid get index will get expection")
    {
        auto bs = Bitset(1);

        EXPECT_THROWS_AS(bs.get(128), IndexOutRange);
    },

    CASE("Set should at correct position.")
    {
        auto bs = Bitset(1);

        bs.set(2, false);

        EXPECT(false == bs.get(2));

        bs.set(3, true);

        EXPECT(true == bs.get(3));

        // others should not be affacted
        EXPECT(false == bs.get(4));

        bs.set(63, true);

        EXPECT(true == bs.get(63));


    },

    CASE("Invalid set index will get exception")
    {
        auto bs = Bitset(1);

        EXPECT_THROWS_AS(bs.get(120), IndexOutRange);
    },

    CASE("Reset can set all bits to true.")
    {
        auto bs = Bitset(2);

        bs.reset(true);

        for (auto i = 0; i < bs.size(); i++)
        {
            EXPECT(true == bs.get(i));
        }
    },

    CASE("Reset default set all to false.")
    {
      auto bs = Bitset(2);

      bs.reset(true);

      bs.reset(false);

      for (auto i = 0; i < bs.size(); i++)
      {
          EXPECT(false == bs.get(i));
      }
    },

    CASE("Resize should not destroy current states.")
    {
        auto bs = Bitset(1);

        bs.reset(true);

        // we have 2 mask items
        bs.resize(128);

        // first 64 should be true by 1st reset
        for (auto i = 0; i < 64; i++)
        {
            EXPECT(true == bs.get(i));
        }

        // others should be false by default
        for (auto i = 64; i < bs.size(); i++)
        {
            EXPECT(false == bs.get(i));
        }
    },

    CASE("Default iterator object will navigate from beginning to the end.")
    {
      auto bs = Bitset(1);

      //
      auto counter = 0;

      auto iter_obj = bs.empty_iter_obj();

      while (!bs.is_end(iter_obj))
      {
        counter++;

        bs.set(bs.empty_index(iter_obj), true);
      }

      // go through all the bits
      EXPECT(64 == counter);

      // then all the bit should be true
      for (auto i = 0; i < bs.size(); i++)
      {
        EXPECT(true == bs.get(i));
      }

      // then iterate again, counter should be 0
      counter = 0;

      iter_obj = bs.empty_iter_obj();

      while (!bs.is_end(iter_obj))
      {
        counter++;
      }

      EXPECT(0 == counter);
    },
};


int main(int argc, char* argv[])
{
  return lest::run(specification, argc, argv);
}
