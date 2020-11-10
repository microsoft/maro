#include "lest.hpp"
#include "../snapshotlist.h"

using namespace std;
using namespace maro::backends::raw;

const lest::test specification[] =
    {
        CASE("Max size can only set for 1 time."){
            auto ss = SnapshotList();

// no snapshots
EXPECT(0 == ss.size());
EXPECT(0 == ss.max_size());

ss.set_max_size(10);

EXPECT(0 == ss.size());
EXPECT(10 == ss.max_size());

ss.set_max_size(20);

EXPECT(0 == ss.size());
EXPECT(10 == ss.max_size());
}
,

    CASE("Max size should not be 0.")
{
  auto ss = SnapshotList();

  EXPECT_THROWS_AS(ss.set_max_size(0), InvalidSnapshotSize);

  // use query and take_snapshot before set_max_size will cause exception too

  auto ats = AttributeStore();

  EXPECT_THROWS_AS(ss.take_snapshot(0, &ats), InvalidSnapshotSize);
}
,

    CASE("Take snapshot without exist tick, no over-write.")
{
  auto ats = AttributeStore();

  ats.add_nodes(0, 0, 10, 0, 1);

  // set value for validation
  {
    auto &attr = ats(0, 0, 0, 0);

    attr = 111;

    auto &attr2 = ats(0, 9, 0, 0);

    attr2 = 999;
  }

  auto ss = SnapshotList();

  ss.set_max_size(2);

  ss.take_snapshot(0, &ats);

  // check internal states
  // NOTE: these state only available under debug mode
  {
    size_t empty_index, empty_length;

    tie(empty_index, empty_length) = ss.empty_states();

    EXPECT(0 == empty_length);
    EXPECT(0 == empty_index);

    auto end_index = ss.end_index();

    EXPECT(10 == end_index);
  }

  {
    auto &a = ss(0, 0, 0, 0, 0);

    EXPECT(111 == a.get_int());
  }

  {
    auto &a = ss(0, 0, 9, 0, 0);

    EXPECT(999 == a.get_int());
  }

  // take 2nd snapshot

  // do something on attribute store to see if it is correct in snapshot list.

  // remove 2nd node
  ats.remove_node(0, 1, 0, 1);

  // change values
  {
    auto &a1 = ats(0, 0, 0, 0);

    a1 = 1111;

    auto &a2 = ats(0, 9, 0, 0);

    a2 = 9999;
  }

  ats.arrange();

  ss.take_snapshot(1, &ats);

  // check attribute values at different tick
  {
    auto &a = ss(0, 0, 0, 0, 0);

    EXPECT(111 == a.get_int());
  }

  {
    auto &a = ss(0, 0, 9, 0, 0);

    EXPECT(999 == a.get_int());
  }

  {
    auto &a = ss(1, 0, 0, 0, 0);

    EXPECT(1111 == a.get_int());
  }

  {
    auto &a = ss(1, 0, 9, 0, 0);

    EXPECT(9999 == a.get_int());
  }

  // invalid tick will return nan
  {
    auto &a = ss(2, 0, 0, 0, 0);

    EXPECT(true == a.is_nan());
  }

  // invalid index will return nan
  {
    auto &a = ss(1, 0, 10, 0, 0);

    EXPECT(true == a.is_nan());
  }

  // check internal states
  {
    size_t empty_index, empty_length;

    tie(empty_index, empty_length) = ss.empty_states();

    // these 2 fields will not changed, as we do not have over-write here
    EXPECT(0 == empty_length);
    EXPECT(0 == empty_index);

    auto end_index = ss.end_index();

    // we removed 1 node at 2nd tick
    // so we have: 10 + 9 left
    EXPECT(19 == end_index);
  }
}
,

    CASE("Take snapshot with exist tick, no over-write.")
{
  /*
  NOTE:

  Take snapshot for same tick without over-write means that all operation is at the end of snapshot list attribute store.

  */

  auto ats = AttributeStore();

  ats.add_nodes(0, 0, 10, 0, 1);

  // set value for validation
  {
    auto &attr = ats(0, 0, 0, 0);

    attr = 111;

    auto &attr2 = ats(0, 9, 0, 0);

    attr2 = 999;
  }

  auto ss = SnapshotList();

  ss.set_max_size(2);

  // take snapshot for tick 0 (1st time)
  ss.take_snapshot(0, &ats);

  // change the value for next time
  {
    auto &a = ats(0, 0, 0, 0);

    a = 222;
  }

  // check internal state of 1st snapshot
  {
    size_t empty_index, empty_length;

    tie(empty_index, empty_length) = ss.empty_states();

    EXPECT(0 == empty_length);
    EXPECT(0 == empty_index);

    auto end_index = ss.end_index();

    EXPECT(10 == end_index);
  }

  // take snapshot for tick 0 (2nd time)
  // this should delete last one
  ss.take_snapshot(0, &ats);

  // check if the value if latest one
  {
    auto &a = ss(0, 0, 0, 0, 0);

    EXPECT(222 == a.get_int());
  }

  // another should keep same
  {
    auto &a = ss(0, 0, 9, 0, 0);

    EXPECT(999 == a.get_int());
  }

  // as we do not change anything, so internal state should not change
  {
    size_t empty_index, empty_length;

    tie(empty_index, empty_length) = ss.empty_states();

    EXPECT(0 == empty_length);
    EXPECT(0 == empty_index);

    auto end_index = ss.end_index();

    EXPECT(10 == end_index);
  }

  // take snapshot for another tick
  ss.take_snapshot(1, &ats);

  // since we only support take snapshot for exist tick if this tick if same as last one,
  // so if we take snapshot for tick 0 here, should cause exception
  EXPECT_THROWS_AS(ss.take_snapshot(0, &ats), InvalidSnapshotTick);
}
,

    CASE("Take snapshot for exist tick, with over-write.")
{
  /*
  NOTE:
    Take snapshot for exist tick with over-write means that operations occur right front of the empty slot area.

    There are 2 case:
    1. last one has enough space to hold new one
    2. last one has no enough space to hold new one
  */

  auto ats = AttributeStore();

  ats.add_nodes(0, 0, 10, 0, 1);

  // set value for validation
  {
    auto &attr = ats(0, 0, 0, 0);

    attr = 111;

    auto &attr2 = ats(0, 9, 0, 0);

    attr2 = 999;
  }

  auto ss = SnapshotList();

  ss.set_max_size(2);

  // normal process
  ss.take_snapshot(0, &ats);
  ss.take_snapshot(1, &ats);

  // this will over-write 1st one (tick 0)
  ss.take_snapshot(2, &ats);

  // over-write tick 0, here will not have empty slots left, as the nodes number node changed
  {
    size_t empty_index, empty_length;

    tie(empty_index, empty_length) = ss.empty_states();

    EXPECT(0 == empty_length);

    // empty index will be changed to point to tick 1
    EXPECT(10 == empty_index);

    auto end_index = ss.end_index();

    // we have 2 snapshots
    EXPECT(20 == end_index);
  }

  EXPECT(2 == ss.size());

  // then any attribute for tick 0 should be NAN
  {
    auto &a = ss(0, 0, 0, 0, 0);

    EXPECT(true == a.is_nan());
  }

  {
    auto &a = ss(0, 0, 9, 0, 0);

    EXPECT(true == a.is_nan());
  }

  // increase the size, then take snapshot, it will erase current tick (2), and append to the end, as tick 2 in snapshot is shorter
  ats.add_nodes(0, 0, 20, 0, 1);

  // change the value
  {
    auto &a1 = ats(0, 0, 0, 0);

    a1 = 1111;

    auto &a2 = ats(0, 9, 0, 0);

    a2 = 9999;

    auto &a3 = ats(0, 19, 0, 0);

    a3 = 19191919;
  }

  ss.take_snapshot(2, &ats);

  EXPECT(2 == ss.size());

  // validate value first
  {
    auto &a = ss(2, 0, 0, 0, 0);

    EXPECT(1111 == a.get_int());

    auto &a2 = ss(2, 0, 9, 0, 0);

    EXPECT(9999 == a2.get_int());

    auto &a3 = ss(2, 0, 19, 0, 0);

    EXPECT(19191919 == a3.get_int());
  }

  {
    size_t empty_index, empty_length;

    tie(empty_index, empty_length) = ss.empty_states();

    // we have 10 empty slots (length of erased snapshot)
    EXPECT(10 == empty_length);

    // we erased old snapshot of tick 2 (length==10), so index will move to front
    EXPECT(0 == empty_index);

    auto end_index = ss.end_index();

    // we have 10 empty slot, 10 for tick 1, 20 for tick 2
    EXPECT(10 + 10 + 20 == end_index);
  }

  // resize nodes number that same as empty slot, then take snapshot for tick 2, will delete 20 from the end, and use empty slots again
  // , and the end_index will be update too

  // remove 10 nodes
  for (auto i = 10; i < 20; i++)
  {
    ats.remove_node(0, i, 0, 1);
  }

  ats.arrange();

  EXPECT(10 == ats.size());
  EXPECT(10 == ats.last_index());

  ss.take_snapshot(2, &ats);

  {
    size_t empty_index, empty_length;

    tie(empty_index, empty_length) = ss.empty_states();

    // all empty slots should be used
    EXPECT(0 == empty_length);

    // empty index will be changed to point to tick 1
    EXPECT(10 == empty_index);

    auto end_index = ss.end_index();

    // we have totally 20 attributes for 2 ticks
    EXPECT(20 == end_index);
  }

  // validate value again to see if value correct
  {
    auto &a = ss(2, 0, 0, 0, 0);

    EXPECT(1111 == a.get_int());

    auto &a2 = ss(2, 0, 9, 0, 0);

    EXPECT(9999 == a2.get_int());

    // this should be NAN, as we delete 20nd node
    auto &a3 = ss(2, 0, 19, 0, 0);

    EXPECT(true == a3.is_nan());
  }
}
,

    CASE("Pass null ptr without set frame before, will cause exception.")
{
  auto ss = SnapshotList();

  ss.set_max_size(2);

  EXPECT_THROWS_AS(ss.take_snapshot(0, nullptr), SnapshotInvalidFrameState);
}
,

    CASE("Pass null ptr with set frame before, will use frame's attributes to to take snapshot")
{
  auto frame = Frame();

  auto n1 = frame.new_node("n1", 10);
  auto a1 = frame.new_attr(n1, "a1", AttrDataType::AINT, 10);

  frame.setup();

  {
    auto &aa1 = frame(0, a1, 0);

    aa1 = 1111;

    auto &aa2 = frame(9, a1, 9);

    aa2 = 9999;
  }

  auto ss = SnapshotList();

  ss.set_frame(&frame);

  ss.set_max_size(10);

  // default attribute store is nullptr
  EXPECT_NO_THROW(ss.take_snapshot(0));

  // validate values
  {
    auto &aa1 = ss(0, n1, 0, a1, 0);

    EXPECT(1111 == aa1.get_int());

    auto &aa2 = ss(0, n1, 9, a1, 9);

    EXPECT(9999 == aa2.get_int());
  }
}
,

    CASE("Query shape should take max slot number")
{
  // NOTE: query need frame
  auto frame = Frame();

  auto n1 = frame.new_node("n1", 10);
  auto a11 = frame.new_attr(n1, "a1", AttrDataType::AINT, 10);
  auto a12 = frame.new_attr(n1, "a2", AttrDataType::AINT, 1);

  EXPECT(a11 == 0);
  EXPECT(a12 == 1);

  auto n2 = frame.new_node("n2", 5);
  auto a21 = frame.new_attr(n2, "a2", AttrDataType::AINT, 1);

  // NOTE: must setup before accessing
  frame.setup();

  // set values for later validation
  {
    auto &a1 = frame(0, a11, 0);
    a1 = 1111;

    auto &a2 = frame(0, a11, 9);
    a2 = 2222;

    auto &a3 = frame(9, a11, 9);
    a3 = 3333;

    auto &a4 = frame(9, a12, 0);
    a4 = 4444;

    auto &a5 = frame(0, a21, 0);
    a5 = 5555;

    auto &a6 = frame(4, a21, 0);
    a6 = 6666;
  }

  auto ticks = vector<INT>{};
  auto nodes = vector<NODE_INDEX>{};
  auto attrs = vector<IDENTIFIER>{a11, a12};

  auto ss = SnapshotList();
  ss.set_max_size(10);

  //  query without set frame will cause exception
  EXPECT_THROWS_AS(ss.prepare(n1, nullptr, 0, nullptr, 0, &attrs[0], attrs.size()), SnapshotInvalidFrameState);

  ss.set_frame(&frame);

  ss.take_snapshot(0);

  // values for tick 1
  {
    auto &a1 = frame(4, a21, 0);
    a1 = 7777;
  }

  ss.take_snapshot(1);

  // do query
  auto shape = ss.prepare(n1, nullptr, 0, nullptr, 0, &attrs[0], attrs.size());

  EXPECT(shape.attr_number == attrs.size());
  EXPECT(shape.max_node_number == 10);
  EXPECT(shape.tick_number == 2);
  EXPECT(shape.max_slot_number == 10);

  // prepare a large enough list to hold result
  auto result = vector<ATTR_FLOAT>();
  result.resize(shape.attr_number * shape.max_node_number * shape.tick_number * shape.max_slot_number);

  // NOTE: we should give result a default value, as query will not set value for nan
  for (auto i = 0; i < result.size(); i++)
  {
    result[i] = NAN;
  }

  ss.query(&result[0], shape);

  //validate result
  EXPECT(1111 == result[0]);
  EXPECT(2222 == result[9]);

  // a12 has 1 slot, so others will be padding value
  EXPECT(true == isnan(result[11]));
}
,

    CASE("Query after reset, should return default value.")
{
  auto frame = Frame();

  auto n1 = frame.new_node("n1", 5);
  auto a1 = frame.new_attr(n1, "a1", AttrDataType::AINT, 1);

  frame.setup();

  auto ss = SnapshotList();
  ss.set_max_size(2);

  ss.set_frame(&frame);

  ss.take_snapshot(0);

  auto attrs = vector<IDENTIFIER>{a1};

  {
    auto shape = ss.prepare(n1, nullptr, 0, nullptr, 0, &attrs[0], attrs.size());

    auto result = vector<ATTR_FLOAT>();

    result.resize(shape.attr_number * shape.max_node_number * shape.max_slot_number * shape.tick_number);

    ss.query(&result[0], shape);

    EXPECT(0 == result[0]);
  }

  frame.reset();
  ss.reset();

  {
    auto shape = ss.prepare(n1, nullptr, 0, nullptr, 0, &attrs[0], attrs.size());
    auto result = vector<ATTR_FLOAT>();

    result.resize(shape.attr_number * shape.max_node_number * shape.max_slot_number * shape.tick_number);
   
   if(result.size() > 0)
   {
    ss.query(&result[0], shape);

    EXPECT(0 == result[0]);
   }
  }
}
}
;

int main(int argc, char *argv[])
{
  return lest::run(specification, argc, argv);
}
