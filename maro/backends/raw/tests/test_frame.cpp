#include "lest.hpp"
#include "../frame.h"


using namespace std;
using namespace maro::backends::raw;

const lest::test specification[] =
{
  CASE("New node/attr should return sequential id.")
  {
    auto frame = Frame();

    auto node1_id = frame.new_node("node1", 10);

    EXPECT(0ULL == node1_id);

    auto node2_id = frame.new_node("node2", 10);

    EXPECT(1ULL == node2_id);

    auto attr1_id = frame.new_attr(node1_id, "attr1", AttrDataType::AINT, 5);

    EXPECT(0ULL == attr1_id);
  },

  CASE("Default value will be 0 after setup")
  {
    auto frame = Frame();

    auto node1_id = frame.new_node("node1", 10);
    auto attr1_id = frame.new_attr(node1_id, "attr1", AttrDataType::AINT, 5);

    frame.setup();

    for (auto node_index = 0; node_index < 10; node_index++)
    {
      for (auto slot = 0; slot < 5; slot++)
      {
        auto& attr = frame(node_index, attr1_id, slot);

        EXPECT(0 == attr.get_int());
      }
    }

    // check if setting can be save
    auto& a1 = frame(0, attr1_id, 0);
    a1 = 10;

    auto& a2 = frame(0, attr1_id, 0);

    EXPECT(10 == a2.get_int());
  },

  CASE("Getter before setup will cause exception.")
  {
    auto frame = Frame();

    EXPECT_THROWS_AS(frame(0, 0, 0), BadAttributeIdentifier);
  },

  CASE("Adding attribute or setting slot before node will cause exception")
  {
    auto frame = Frame();

    EXPECT_THROWS_AS(frame.new_attr(0, "a1", AttrDataType::AINT, 10), BadNodeIdentifier);

    EXPECT_THROWS_AS(frame.set_attr_slot(0, 10), BadAttributeIdentifier);
  },

  CASE("Adding additional nodes will not corrupt existings, and should have same attributes as defined.")
  {
    auto frame = Frame();

    auto n1 = frame.new_node("node1", 10);

    auto a1 = frame.new_attr(n1, "a1", AttrDataType::AINT, 5);
    auto a2 = frame.new_attr(n1, "a2", AttrDataType::AFLOAT, 1);

    frame.setup();

    // set a value as validate flag
    {
      // last slot of a1 for 1st node
      auto& aa = frame(0, a1, 4);

      aa = 110;
    }

    // validate last setting
    {
      auto& aa = frame(0, a1, 4);

      EXPECT(110 == aa.get_int());
    }

    // access by node index large than defined will cause error
    {
      EXPECT_THROWS_AS(frame(19, a1, 4), BadNodeIndex);
    }

    // 10 additional nodes
    frame.add_node(n1, 10);

    // existing value should not be affected.
    {
      auto& aa = frame(0, a1, 4);

      EXPECT(110 == aa.get_int());
    }

    // get attribute for additional nodes
    {
      auto& aa = frame(19, a1, 4);

      EXPECT(0 == aa.get_int());
    }
  },

    CASE("Getter for removed nodes will cause exception.")
  {
    auto frame = Frame();

    auto n1 = frame.new_node("node1", 10);
    auto a1 = frame.new_attr(n1, "a1", AttrDataType::AINT, 1);

    frame.setup();

    // set value for 1st node
    {
      auto& aa = frame(0, a1, 0);

      aa = 10;
    }

    // remove 1st node
    {
      frame.remove_node(n1, 0);

      EXPECT_THROWS_AS(frame(0, a1, 0), BadAttributeIndexing);
    }

    // access 2nd should work
    {
      auto& aa = frame(1, a1, 0);

      EXPECT(0 == aa.get_int());
    }
  },

    CASE("Set attribute slots will only affect slots at the tail")
  {
    auto frame = Frame();

    auto n1 = frame.new_node("n1", 10);
    auto a1 = frame.new_attr(n1, "a1", AttrDataType::AINT, 5);

    // setup before accessing
    frame.setup();

    // set value for 5 slots of 1st node
    {
      for (auto i = 0; i < 5; i++)
      {
        auto& aa = frame(0, a1, i);

        aa = i;
      }
    }

    // narrow down the slots will keep slots value
    frame.set_attr_slot(a1, 3);

    {
      for (auto i = 0; i < 3; i++)
      {
        auto& aa = frame(0, a1, i);

        EXPECT(i == aa.get_int());
      }

      // accessing removed ones will cause exception
      EXPECT_THROWS_AS(frame(0, a1, 3), BadAttributeSlotIndex);
      EXPECT_THROWS_AS(frame(0, a1, 4), BadAttributeSlotIndex);
    }


    // extend it
    frame.set_attr_slot(a1, 6);

    {
      for (auto i = 0; i < 3; i++)
      {
        auto& aa = frame(0, a1, i);

        EXPECT(i == aa.get_int());
      }
      
      // extended value will allocate new space, so value will be default
      EXPECT(0 == frame(0, a1, 3).get_int());
      EXPECT(0 == frame(0, a1, 4).get_int());
      EXPECT(0 == frame(0, a1, 5).get_int());
    }
  }
};



int main(int argc, char* argv[])
{
  return lest::run(specification, argc, argv);
}
