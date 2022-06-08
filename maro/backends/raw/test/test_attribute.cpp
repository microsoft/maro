#include <gtest/gtest.h>

#include "../attribute.h"



  using namespace maro::backends::raw;

  // test attribute creation
  TEST(Attribute, Creation) {
    Attribute attr;

    EXPECT_EQ(attr.get_type(), AttrDataType::ACHAR);
    EXPECT_FALSE(attr.is_nan());
    EXPECT_EQ(attr.slot_number, 0);
   
  }

  // test create attribute with other type value.
  TEST(Attribute, CreateWithTypedValue) {
    Attribute attr{ ATTR_UINT(12)};

    EXPECT_EQ(attr.get_type(), AttrDataType::AUINT);
    EXPECT_EQ(attr.get_value<ATTR_UINT>(), 12);
    EXPECT_EQ(attr.slot_number, 0);
    EXPECT_FALSE(attr.is_nan());
  }

  // test is nan case
  TEST(Attribute, CreateWithNan) {
    Attribute attr{ nan("nan")};

    EXPECT_TRUE(attr.is_nan());
  }

