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

      class InvalidSnapshotTick : public exception
      {};

      class InvalidSnapshotSize : public exception
      {};

      class SnapshotQueryNotPrepared : public exception
      {
      };

      class SnapshotQueryNoAttributes : public exception
      {};

      class SnapshotInvalidFrameState: public exception
      {};

      class SnapshotQueryResultPtrNull:public exception
      {};

      /**
      Steps to take snapshot:

      1. if there is an exist tick?
        a. yes: if (exist snapshot area + empty slots) is enough to hold current snapshot?
          I. yes: write from exist area start
          II. no: mark exist snapshot area as empty slots, append current snapshot to the end
        b. no: if reach the limitation?
          I. yes: mark oldest snapshot as empty slots, if remaining empty slots enough to hold current snapshot?
            I). yes: write to oldest area
            II). no: keep these empty slots there, append current snapshot to the end
          II. no: append to the end


      */

      /**

      Steps to query:

      We expect that the result pointer (float*) is a 4d array (numpy array), and length of items for each dimension should be same.


      */

      struct SnapshotResultShape
      {
        /* Following 4 parts used for out-side to construct the result array */
        INT tick_number{ 0 };
        NODE_INDEX max_node_number{ 0 };
        USHORT attr_number{ 0 };
        SLOT_INDEX max_slot_number{ 0 };
      };


      class SnapshotList
      {
        /// <summary>
        /// Object used to hold the parameter that used for query
        /// </summary>
        struct SnapshotQueryParameters
        {
          // for furthur querying, these fields would be changed by prepare function
          IDENTIFIER node_id{ 0 };
          INT* ticks{ nullptr };
          UINT tick_length{ 0 };
          NODE_INDEX* node_indices{ nullptr };
          UINT node_length{ 0 };
          IDENTIFIER* attributes{ nullptr };
          UINT attr_length{ 0 };

          void reset();
        };

        Frame* _frame;

        // tick -> [node_ide, node_index, attr_id, slot_index] -> index in attr store
        //map<INT, map<ULONG, ULONG>> _attr_map;
        map<INT, unordered_map<ULONG, size_t>> _tick_attr_map;

        // Used to hold all the in-memory snapshots
        vector<Attribute> _attr_store;

        // where shall we start to check empty slots to hold latest snapshot
        size_t _first_empty_slot_index{ 0 };
        size_t _empty_slots_length{ 0 };

        // end index of attribute store that been used
        // we append from this point
        size_t _end_index{ 0 };

        // max number of snapshot we should keep in memory
        USHORT _max_size{ 0 };

        // current number of snapshot
        USHORT _cur_snapshot_num{ 0 };

        // last tick that take snapshot
        // used to track same tick over-writing (for last operation only)
        INT _last_tick{ -1 };


        map<INT, size_t> _tick2index_map; // tick -> start index
        map<INT, size_t> _tick2size_map; // size of each tick

        Attribute _defaultAttr = Attribute(NAN);

        bool _is_prepared{ false };
        SnapshotQueryParameters _query_parameters;

      public:

        void set_frame(Frame* frame);

        void set_max_size(USHORT max_size);

        /// <summary>
        /// Take snapshot for current frame, this function will arrange current frame
        /// </summary>
        /// <param name="tick"></param>
        void take_snapshot(INT tick, AttributeStore* frame_attr_store);

        Attribute& operator() (INT tick, IDENTIFIER node_id, NODE_INDEX node_index, IDENTIFIER attr_id, SLOT_INDEX slot_index);

        USHORT size();
        USHORT max_size();


        // prepare for querying use passed parameters, this method will correct input info, and generate an parameter object for next query
        SnapshotResultShape prepare(IDENTIFIER node_id, INT ticks[], UINT tick_length,
          NODE_INDEX node_indices[], UINT node_length, IDENTIFIER attributes[], UINT attr_length);

        // do query using parameters from last prepare invoking, cause exception if without prepare calling
        // it will reset prepare state to false, so DO make sure prepare for each querying
        void query(QUERING_FLOAT* result, SnapshotResultShape shape);


#ifdef _DEBUG
        pair<size_t, size_t> empty_states();

        size_t end_index();
#endif

      private:
        void append_to_end(AttributeStore* frame_attr_store, INT tick);
        void write_to_empty_slots(AttributeStore* frame_attr_store, INT tick);
        inline void ensure_max_size();
      };
    }
  }
}

#endif // !_MARO_BACKENDS_RAW_SNAPSHOTLIST
