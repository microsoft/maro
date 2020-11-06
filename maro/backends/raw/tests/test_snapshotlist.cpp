#include "lest.hpp"
#include "../snapshotlist.h"


using namespace std;
using namespace maro::backends::raw;

const lest::test specification[] =
{
  CASE("Max size can only set for 1 time.")
  {
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
  },

  CASE("Max size should not be 0.")
  {
    auto ss = SnapshotList();

    EXPECT_THROWS_AS(ss.set_max_size(0), InvalidSnapshotSize);

    // use query and take_snapshot before set_max_size will cause exception too

    auto ats = AttributeStore();

    EXPECT_THROWS_AS(ss.take_snapshot(0, ats), InvalidSnapshotSize);
},

CASE("Take snapshot without exist tick, no over-write.")
{
  auto ats = AttributeStore();

  ats.add_nodes(0, 0, 10, 0, 1);

  // set value for validation
  {
    auto& attr = ats(0, 0, 0, 0);

    attr = 111;

    auto& attr2 = ats(0, 9, 0, 0);

    attr2 = 999;
  }

    auto ss = SnapshotList();

    ss.set_max_size(2);

    ss.take_snapshot(0, ats);

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
      auto& a = ss(0, 0, 0, 0, 0);

      EXPECT(111 == a.get_int());
    }

    {
      auto& a = ss(0, 0, 9, 0, 0);

      EXPECT(999 == a.get_int());
    }

    // take 2nd snapshot

    // do something on attribute store to see if it is correct in snapshot list.

    // remove 2nd node
    ats.remove_node(0, 1, 0, 1);

    // change values
    {
      auto& a1 = ats(0, 0, 0, 0);

      a1 = 1111;

      auto& a2 = ats(0, 9, 0, 0);

      a2 = 9999;
    }

    ats.arrange();

    ss.take_snapshot(1, ats);

    // check attribute values at different tick
    {
      auto& a = ss(0, 0, 0, 0, 0);

      EXPECT(111 == a.get_int());
    }

    {
      auto& a = ss(0, 0, 9, 0, 0);

      EXPECT(999 == a.get_int());
    }

    {
      auto& a = ss(1, 0, 0, 0, 0);

      EXPECT(1111 == a.get_int());
    }

    {
      auto& a = ss(1, 0, 9, 0, 0);

      EXPECT(9999 == a.get_int());
    }

    // invalid tick will return nan
    {
      auto& a = ss(2, 0, 0, 0, 0);

      EXPECT(true == a.is_nan());
    }

    // invalid index will return nan
    {
      auto& a = ss(1, 0, 10, 0, 0);

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

},

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
    auto& attr = ats(0, 0, 0, 0);

    attr = 111;

    auto& attr2 = ats(0, 9, 0, 0);

    attr2 = 999;
  }


  auto ss = SnapshotList();

  ss.set_max_size(2);


  // take snapshot for tick 0 (1st time)
  ss.take_snapshot(0, ats);

  // change the value for next time
  {
    auto& a = ats(0, 0, 0, 0);

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
  ss.take_snapshot(0, ats);


  // check if the value if latest one
  {
    auto& a = ss(0, 0, 0, 0, 0);

    EXPECT(222 == a.get_int());
  }

  // another should keep same
  {
    auto& a = ss(0, 0, 9, 0, 0);

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
  ss.take_snapshot(1, ats);

  // since we only support take snapshot for exist tick if this tick if same as last one,
  // so if we take snapshot for tick 0 here, should cause exception
  EXPECT_THROWS_AS(ss.take_snapshot(0, ats), InvalidSnapshotTick);
},

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
    auto& attr = ats(0, 0, 0, 0);

    attr = 111;

    auto& attr2 = ats(0, 9, 0, 0);

    attr2 = 999;
  }

  auto ss = SnapshotList();

  ss.set_max_size(2);

  // normal process
  ss.take_snapshot(0, ats);
  ss.take_snapshot(1, ats);

  // this will over-write 1st one (tick 0)
  ss.take_snapshot(2, ats);

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
    auto& a = ss(0, 0, 0, 0, 0);

    EXPECT(true == a.is_nan());
  }

  {
    auto& a = ss(0, 0, 9, 0, 0);

    EXPECT(true == a.is_nan());
  }

  // increase the size, then take snapshot, it will erase current tick (2), and append to the end, as tick 2 in snapshot is shorter
  ats.add_nodes(0, 0, 20, 0, 1);

  // change the value
  {
    auto& a1 = ats(0, 0, 0, 0);

    a1 = 1111;

    auto& a2 = ats(0, 9, 0, 0);

    a2 = 9999;

    auto& a3 = ats(0, 19, 0, 0);

    a3 = 19191919;
  }

  ss.take_snapshot(2, ats);

  EXPECT(2 == ss.size());

  // validate value first
  {
    auto& a = ss(2, 0, 0, 0, 0);

    EXPECT(1111 == a.get_int());

    auto& a2 = ss(2, 0, 9, 0, 0);

    EXPECT(9999 == a2.get_int());

    auto& a3 = ss(2, 0, 19, 0, 0);

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

  ss.take_snapshot(2, ats);

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
    auto& a = ss(2, 0, 0, 0, 0);

    EXPECT(1111 == a.get_int());

    auto& a2 = ss(2, 0, 9, 0, 0);

    EXPECT(9999 == a2.get_int());

    // this should be NAN, as we delete 20nd node
    auto& a3 = ss(2, 0, 19, 0, 0);

    EXPECT(true == a3.is_nan());
  }
},

};

int main(int argc, char* argv[])
{
  return lest::run(specification, argc, argv);
}
