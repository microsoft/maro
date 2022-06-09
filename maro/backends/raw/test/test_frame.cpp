#include <iomanip>
#include <array>

#include <gtest/gtest.h>

#include "../common.h"
#include "../frame.h"
#include "../snapshotlist.h"

using namespace maro::backends::raw;


TEST(test, correct) {
  EXPECT_EQ(1, 1);
}

// show how to use frame and snapshot at c++ end
TEST(test, show_case) {
  // a new frame
  Frame frame;

  // add a new node with a name
  auto node_type = frame.add_node("test_node", 1);

  // add an attribute to this node, this is a list attribute, it has different value to change the value
  // NOTE: list means is it dynamic array, that the size can be changed even after setting up
  auto attr_type_1 = frame.add_attr(node_type, "a1", AttrDataType::AUINT, 10, false, true);

  // this is a normal attribute
  // NOTE: list == false means it is a fixed array that cannot change the size after setting up
  auto attr_type_2 = frame.add_attr(node_type, "a2", AttrDataType::AUINT, 2);

  // setup means initialize the frame with node definitions (allocate memory)
  // NOTE: call this method before accessing the attributes
  frame.setup();

  // list and normal attribute have different method to set value
  frame.set_value<ATTR_UINT>(0, attr_type_2, 0, 33554441);
  frame.insert_to_list<ATTR_UINT>(0, attr_type_1, 0, 33554442);

  // but they have same get method
  auto v1 = frame.get_value<ATTR_UINT>(0, attr_type_1, 0);
  auto v2 = frame.get_value<ATTR_UINT>(0, attr_type_2, 0);

  // test with true type
  EXPECT_EQ(v2, 33554441);
  EXPECT_EQ(v1, 33554442);

  // test with query result type
  EXPECT_EQ(QUERY_FLOAT(v2), 3.3554441e+07);
  EXPECT_EQ(QUERY_FLOAT(v1), 3.3554442e+07);

  // snapshot instance
  SnapshotList ss;

  // NOTE: we need following 2 method to initialize the snapshot instance, or accessing will cause exception
  // which frame we will use to copy the values
  ss.setup(&frame);
  // max snapshot it will keep, oldeat one will be delete when reading the limitation
  ss.set_max_size(10);

  // take a snapshot for a tick
  ss.take_snapshot(0);

  // query parameters
  std::array<int, 1> ticks{ 0 };
  std::array<NODE_INDEX, 1> indices{ 0 };
  std::array< ATTR_TYPE, 1> attributes{attr_type_1};

  // we need use the parameter to get how many items we need to hold the results
  auto shape = ss.prepare(node_type, &(ticks[0]), ticks.size(), &(indices[0]), indices.size(), &(attributes[0]), attributes.size());

  auto total = shape.tick_number * shape.max_node_number * shape.max_slot_number * shape.attr_number;

  // then query (the snapshot instance will remember the latest query parameters, so just pass the result array
  QUERY_FLOAT* results = new QUERY_FLOAT[total];

  ss.query(results);

  // 1st slot value of first node
  EXPECT_EQ(results[0], 3.3554442e+07);

  delete[] results;
}
