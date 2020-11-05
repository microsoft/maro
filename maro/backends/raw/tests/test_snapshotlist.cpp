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

    EXPECT_THROWS_AS(ss.query(nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0), InvalidSnapshotSize);
},

CASE("Take snapshot without exist tick.")
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

  {
    auto ss = SnapshotList();

    ss.set_max_size(2);

    ss.take_snapshot(0, ats);

    auto& a1 = ss(0, 0, 0, 0, 0);

    EXPECT(111 == a1.get_int());
   }

},

CASE("Take snapshot with exist tick.")
{

},

CASE("Take snapshot without over-write.")
{

},

CASE("Take snapshot with over-write.")
{

},

};

int main(int argc, char* argv[])
{
  return lest::run(specification, argc, argv);
}
