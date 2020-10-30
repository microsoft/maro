#ifndef _MARO_BACKENDS_RAW_SNAPSHOTLIST
#define _MARO_BACKENDS_RAW_SNAPSHOTLIST

#include <map>
#include <vector>

#include "common.h"
#include "attribute.h"
#include "frame.h"

namespace maro
{
  namespace backends
  {
    namespace raw
    {

      /**
      When taking snapshot, ss will arrange current frame first, then check if meet the capacity:
      1. no: copy current frame attributes to the end, copy and update the attribute mapping to related tick
      2. yes: check if the oldest snapshot long enough to hold current frame.
        a. yes: copy attribute into oldest place, mark leaving slots as un-used
        b. no: check slots that marked as un-used, copy into these slots, or allocate more to hold

      */

      template<typename A, typename V>
      class AttrMap : public map<A, V> {};

      template<typename T, typename A, typename V>
      class TickAttrMap : public map<T, AttrMap<A, V>> {};



      class SnapshotList
      {
        // reference to current frame
        Frame* _cur_frame{ nullptr };

        // tick -> [node_ide, node_index, attr_id, slot_index] -> index in attr store
        //map<INT, map<ULONG, ULONG>> _attr_map;
        TickAttrMap<INT, ULONG, ULONG> _tick_attr_map;

        // Used to hold all the in-memory snapshots
        vector<Attribute> _attr_store;

        // where shall we start to check empty slots to hold latest snapshot
        size_t _first_empty_slot_index;

        // end index of attribute store that been used
        size_t _end_index;

      public:
        /// <summary>
        /// Take snapshot for current frame, this function will arrange current frame
        /// </summary>
        /// <param name="tick"></param>
        void take_snapshot(INT tick);

        /// <summary>
        /// Query
        /// </summary>
        /// <param name="result"></param>
        /// <param name="node_id"></param>
        /// <param name=""></param>
        /// <param name="node_length"></param>
        /// <param name="attributes"></param>
        /// <param name="attr_length"></param>
        void query(QUERING_FLOAT* result, IDENTIFIER node_id, INT ticks[], UINT tick_length,
          NODE_INDEX node_indices[], UINT node_length, IDENTIFIER attributes, UINT attr_length);
      };
    }
  }
}

#endif // !_MARO_BACKENDS_RAW_SNAPSHOTLIST
