// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef _MARO_BACKENDS_RAW_SNAPSHOTLIST
#define _MARO_BACKENDS_RAW_SNAPSHOTLIST

#include <map>
#include <vector>
#include <iostream>
#include <fstream>

#include "common.h"
#include "attribute.h"
#include "frame.h"

namespace maro
{
  namespace backends
  {
    namespace raw
    {
      /// <summary>
      /// Tick not supported, like negative tick
      /// </summary>
      class InvalidSnapshotTick : public exception
      {
      };

      /// <summary>
      /// Snapshot list max size is 0
      /// </summary>
      class InvalidSnapshotSize : public exception
      {
      };

      /// <summary>
      /// Query without call prepare function
      /// </summary>
      class SnapshotQueryNotPrepared : public exception
      {
      };

      /// <summary>
      /// Attribute not exist when querying
      /// </summary>
      class SnapshotQueryNoAttributes : public exception
      {
      };

      /// <summary>
      /// Frame not set before operations
      /// </summary>
      class SnapshotInvalidFrameState : public exception
      {
      };

      /// <summary>
      /// Array pointer is nullptr
      /// </summary>
      class SnapshotQueryResultPtrNull : public exception
      {
      };

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
        INT tick_number{0};
        NODE_INDEX max_node_number{0};
        USHORT attr_number{0};
        SLOT_INDEX max_slot_number{0};
      };

      class SnapshotList
      {
        /// <summary>
        /// Object used to hold the parameter that used for query
        /// </summary>
        struct SnapshotQueryParameters
        {
          // for furthur querying, these fields would be changed by prepare function
          IDENTIFIER node_id{0};
          INT *ticks{nullptr};
          UINT tick_length{0};
          NODE_INDEX *node_indices{nullptr};
          UINT node_length{0};
          IDENTIFIER *attributes{nullptr};
          UINT attr_length{0};

          void reset();
        };

        Frame *_frame;

        // tick -> index of mapping  -> [node_ide, node_index, attr_id, slot_index] -> index in attr store
        map<INT, size_t> _tick_attr_map;

        // Used to hold all the in-memory snapshots
        vector<Attribute> _attr_store;

        // where shall we start to check empty slots to hold latest snapshot
        size_t _first_empty_slot_index{0};
        size_t _empty_slots_length{0};

        // end index of attribute store that been used
        // we append from this point
        size_t _end_index{0};

        // max number of snapshot we should keep in memory
        USHORT _max_size{0};

        // current number of snapshot
        USHORT _cur_snapshot_num{0};

        // last tick that take snapshot
        // used to track same tick over-writing (for last operation only)
        INT _last_tick{-1};

        map<INT, size_t> _tick2index_map; // tick -> start index
        map<INT, size_t> _tick2size_map;  // size of each tick

        Attribute _defaultAttr = Attribute(NAN);

        bool _is_prepared{false};
        SnapshotQueryParameters _query_parameters;

        vector<unordered_map<ULONG, size_t>> _mappings;

      public:
        /// <summary>
        /// Set frame we need for taking snapshot
        /// </summary>
        /// <param name="frame">Frame we need</param>
        void set_frame(Frame *frame);

        /// <summary>
        /// Set max size of snapshots
        /// </summary>
        /// <param name="max_size">Max size</param>
        void set_max_size(USHORT max_size);

        /// <summary>
        /// Take snapshot for current frame, this function will arrange current frame.
        ///
        /// NOTE:
        /// This function support taking snapshot without frame
        /// </summary>
        /// <param name="tick">Tick of current snapshot</param>
        /// <param name="frame_attr_store">Attributes store used to take snapshot</param>
        void take_snapshot(INT tick, AttributeStore *frame_attr_store = nullptr);

        /// <summary>
        /// Get an attribute
        /// </summary>
        /// <param name="tick">Tick of attribute</param>
        /// <param name="node_id">Id of node</param>
        /// <param name="node_index">Index of node</param>
        /// <param name="attr_id">Id of attribute</param>
        /// <param name="slot_index">Index of slot</param>
        /// <returns>Specified attribute, return NAN attribute if specified not exist</returns>
        Attribute &operator()(INT tick, IDENTIFIER node_id, NODE_INDEX node_index, IDENTIFIER attr_id, SLOT_INDEX slot_index);

        /// <summary>
        /// size of current snapshots
        /// </summary>
        /// <returns>Size of current snapshots</returns>
        USHORT size();

        /// <summary>
        /// Max size of snapshot list
        /// </summary>
        /// <returns>Max size</returns>
        USHORT max_size();

        /// <summary>
        /// Reset current snapshots
        /// </summary>
        void reset();

        /// <summary>
        /// Dump current snapshot into specified folder
        /// </summary>
        /// <param name="path">Folder to place the files</param>
        void dump(string path);

        /// <summary>
        /// Get ticks of current snapshot list
        /// </summary>
        /// <param name="result">Pointer to hold result</param>
        void get_ticks(INT *result);

        /// <summary>
        /// Prepare for querying, used to get shape of result.
        /// </summary>
        /// <param name="node_id">Id of node to query</param>
        /// <param name="ticks">Tick list to query</param>
        /// <param name="tick_length">Length of tick list</param>
        /// <param name="node_indices">Node index list to query</param>
        /// <param name="node_length">Length of node index list</param>
        /// <param name="attributes">Attribute id list to query</param>
        /// <param name="attr_length">Lenght of attribute list</param>
        /// <returns>Shape of expected result</returns>
        SnapshotResultShape prepare(IDENTIFIER node_id, INT ticks[], UINT tick_length,
                                    NODE_INDEX node_indices[], UINT node_length, IDENTIFIER attributes[], UINT attr_length);

        /// <summary>
        /// Qeury with shape from prepare function
        /// </summary>
        /// <param name="result">Array to hold result, the size must larger or equal to shape size</param>
        /// <param name="shape">Shape from prepare function</param>
        void query(ATTR_FLOAT *result, SnapshotResultShape shape);

#ifdef _DEBUG
        pair<size_t, size_t> empty_states();

        size_t end_index();
#endif

      private:
        // helper function to append attributes of current frame to the end
        void append_to_end(AttributeStore *frame_attr_store, INT tick);

        // helper function to place attributes of current frame to empty slots
        void write_to_empty_slots(AttributeStore *frame_attr_store, INT tick);

        // helper function to ensure the max size is correct
        inline void ensure_max_size();

        // function to write attribute to file stream
        inline void write_attribute(ofstream &file, INT tick, IDENTIFIER node_id, NODE_INDEX node_index, IDENTIFIER attr_id, SLOT_INDEX slot_index);

        // prepare memory before taking snapshot (only for first time), to avoid to much allocation operation
        inline void prepare_memory();

        // copy attributes from store to specified position
        void copy_from_attr_store(AttributeStore *frame_attr_store, INT tick, size_t start_index);
      };
    } // namespace raw
  }   // namespace backends
} // namespace maro

#endif // !_MARO_BACKENDS_RAW_SNAPSHOTLIST
