#include "lest.hpp"
#include "../attributestore.h"


using namespace std;
using namespace maro::backends::raw;

const lest::test specification[] =
{
  CASE("Setup with specified size")
  {
    // TODO: fill later
  },

  CASE("Getter before any adding should cause exception.")
  {
    auto ats = AttributeStore();

    EXPECT_THROWS_AS(ats(0, 0, 0, 0), BadAttributeIndexing);
  },

  CASE("Getter should return reference of attribute.")
  {
    auto ats = AttributeStore();

    ats.setup(10);

    // add 1 node type 1 node, each node contains attribute 0 with 10 slots
    ats.add(0, 1, 0, 10);

    // get 1st slot

    auto& attr = ats(0, 0, 0, 0);

    // get value
    // NOTE: this will change data type of attribute
    attr = 10;

    //
    EXPECT(10 == attr.get_int());

    // get again to see if value changed
    EXPECT(10 == ats(0, 0, 0, 0).get_int());


    // invalid index will cause exception
    EXPECT_THROWS_AS(ats(0, 0, 0, 11), BadAttributeIndexing);
  },

  CASE("Getter with invalid index or id will cause exception.")
  {
    auto ats = AttributeStore();

    ats.add(0, 10, 0, 1);


    // get with invalid node index
    EXPECT_THROWS_AS(ats(0, 10, 0, 0), BadAttributeIndexing);

    // get with invalid node id
    EXPECT_THROWS_AS(ats(1, 0, 0, 0), BadAttributeIndexing);

    // get with invalid slot index
    EXPECT_THROWS_AS(ats(0, 0, 0, 1), BadAttributeIndexing);

    // get with invalid attribute id
    EXPECT_THROWS_AS(ats(0, 0, 1, 0), BadAttributeIndexing);
  },

  CASE("Add will extend the existing space.")
  {
    auto ats = AttributeStore();

    ats.setup(10);

    // add 1 node attribute
    ats.add(0, 5, 0, 1);


    // size should be same as setup specified
    EXPECT(10 == ats.size());

    // 2nd attribute
    ats.add(0, 5, 1, 1);

    // still within the capacity
    EXPECT(10 == ats.size());

    // this will extend internal space
    ats.add(0, 5, 2, 10);

    // the size will be changed (double size of last_index)
    EXPECT(120 == ats.size());
    EXPECT(60 == ats.last_index());
    
  },

  CASE("Add without setup works same.")
  {
    auto ats = AttributeStore();

    ats.add(0, 5, 0, 1);

    EXPECT(5 == ats.last_index());
    EXPECT(10 == ats.size());
  },

  CASE("Remove will cause empty slots in the middle of vector")
  {
    auto ats = AttributeStore();

    ats.setup(10);

    // add 1 attribute for node id 0, it will take 1st 6 slots
    ats.add(0, 2, 0, 3);

    // set value for 2nd attribute of 1st node
    auto& attr = ats(0, 0, 0, 1);

    // update the value
    attr = 10;

    // remove 2nd node
    ats.remove(0, 1, 0, 3);

    // then getter for 2nd node should cause error
    EXPECT_THROWS_AS(ats(0, 1, 0, 1), BadAttributeIndexing);

    // but 1st node's attribute will not be affected.
    attr = ats(0, 0, 0, 1);
    EXPECT(10 == attr.get_int());
  },

  CASE("Remove with invalid parameter will not cause error.")
  {
    auto ats = AttributeStore();

    ats.add(0, 1, 0, 10);

    // remove un-exist node_id
    EXPECT_NO_THROW(ats.remove(1, 0, 0, 10));

    // remove with invalid node index
    EXPECT_NO_THROW(ats.remove(0, 1, 0, 10));

    // remove with invalid attribute id
    EXPECT_NO_THROW(ats.remove(0, 0, 1, 10));

    // remove with invalid attribute slot number;
    EXPECT_NO_THROW(ats.remove(0, 0, 0, 20));
  },

  CASE("Arrange should fill empty slots with attribute at the end.")
  {
    auto ats = AttributeStore();

    ats.add(0, 2, 0, 10);

    // after adding node attributes, last index should be 20
    EXPECT(20 == ats.last_index());

    // set value for last attribute of 2nd node
    auto& attr = ats(0, 1, 0, 9);

    attr = 10;

    // remove 1st node to gen empty slots
    ats.remove(0, 0, 0, 10);

    // arrange should work without exception
    EXPECT_NO_THROW(ats.arrange());

    // last index should be 10, as we will fill 10 empty slots
    EXPECT(10 == ats.last_index());

    // and our last node will be moved to 1st slot, but with updated index
    attr = ats(0, 1, 0, 9);

    // so value should not be changed
    EXPECT(10 == attr.get_int());

    // size will not change too

  },
}
;
int main(int argc, char* argv[])
{
  return lest::run(specification, argc, argv);
}
