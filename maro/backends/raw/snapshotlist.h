// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef _MARO_BACKENDS_RAW_SNAPSHOTLIST_
#define _MARO_BACKENDS_RAW_SNAPSHOTLIST_


#include <map>
#include <vector>
#include <string>
#include <iostream>

#include "common.h"
#include "attribute.h"
#include "node.h"
#include "frame.h"

using namespace std;

namespace maro
{
  namespace backends
  {
    namespace raw
    {
      #define MAX(a, b) a > b ? a : b

      /// <summary>
      /// Shape of current querying.
      /// </summary>
      struct SnapshotQueryResultShape
      {
        // Number of attribute in result.
        USHORT attr_number = 0;

        // Number of ticks in result.
        int tick_number = 0;

        // Number of slot in result, include padding slot.
        SLOT_INDEX max_slot_number = 0;

        // Number of node in result, include padding nodes.
        NODE_INDEX max_node_number = 0;
      };

      /// <summary>
      /// Snapshot list used to hold snapshot of current frame at specified tick.
      /// </summary>
      class SnapshotList
      {
        /// <summary>
        /// Querying parameter from prepare step.
        /// </summary>
        struct SnapshotQueryParameters
        {
          // Is this query for list?
          bool is_list = false;

          // For furthur querying, these fields would be changed by prepare function.
          NODE_TYPE node_type = 0;

          // List of ticks to query.
          int* ticks = nullptr;

          // Number of ticks in tick list.
          UINT tick_length = 0;

          // List of node instance index to query.
          NODE_INDEX* node_indices = nullptr;

          // Node number
          UINT node_length = 0;

          // Attributes to query.
          ATTR_TYPE* attributes = nullptr;

          // Number of attribute to query.
          UINT attr_length = 0;

          // Max slot number in result, for padding.
          SLOT_INDEX max_slot_number = 0;

          /// <summary>
          /// Reset current parameter after querying.
          /// </summary>
          void reset();
        };


      private:
        // Tick and its snapshot frame, we will keep a copy of frame.
        map<int, Frame> _snapshots;

        // Max size of snapshot is memory.
        USHORT _max_size = 0;

        // Current frame that used to copy.
        Frame* _cur_frame;

        // Used to hold parameters from prepare function.
        SnapshotQueryParameters _query_parameters;

        // Is prepare function called?
        bool _is_prepared = false;

        // Default attribute for invalid attribute, for padding.
        Attribute _nan_attr = NAN;

        // Query state for list attribute.
        // NOTE: for list attribute, we only support 1 tick, 1 attribute, 1 node.
        // and node cannot be null. If ticks not provided, then use latest tick.
        void query_for_list(QUERY_FLOAT* result);

        // Query for normal attributes.
        void query_for_normal(QUERY_FLOAT* result);

        // Get attribute from specified tick, this function will not throw exception, it will return a NAN attribute
        // if invalid.
        Attribute& get_attr(int tick, NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX slot_index) noexcept;

        // Make sure currect frame not null.
        inline void ensure_cur_frame();

        // Make sure max size greater than 0.
        inline void ensure_max_size();

        inline void write_attribute(ofstream &file, int tick, NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX slot_index);
      public:
        /// <summary>
        /// Set max size of snapshot in memory.
        /// </summary>
        /// <param name="max_size">Max size to set.</param>
        void set_max_size(USHORT max_size);

        /// <summary>
        /// Setup snapshot list with current frame.
        /// </summary>
        /// <param name="frame">Current frame that used for snapshots.</param>
        void setup(Frame* frame);

        /// <summary>
        /// Take snapshot for specified tick.
        /// </summary>
        /// <param name="ticks">Tick to take snapshot.</param>
        void take_snapshot(int ticks);

        /// <summary>
        /// Current size of snapshots.
        /// </summary>
        /// <returns>Number of current snapshots.</returns>
        UINT size() const noexcept;

        /// <summary>
        /// Get max size of current snapshot list.
        /// </summary>
        /// <returns>Max number of snapshot list.</returns>
        UINT max_size() const noexcept;

        /// <summary>
        /// Reset snapshot list states.
        /// </summary>
        void reset();

        /// <summary>
        /// Dump current snapshots into folder, node will be split into different files.
        /// </summary>
        void dump(string path);

        /// <summary>
        /// Get avaiable ticks from snapshot list.
        /// </summary>
        /// <param name="result">List pointer to hold ticks.</param>
        void get_ticks(int* result) const;

        /// <summary>
        /// Get current max node number for specified node type.
        /// </summary>
        /// <param name="node_type">Target node type.</param>
        /// <returns>Max node number.</returns>
        NODE_INDEX get_max_node_number(NODE_TYPE node_type) const;

        /// <summary>
        /// Prepare for querying.
        /// </summary>
        /// <param name="node_type">Target node type.</param>
        /// <param name="ticks">Ticks to query, leave it as null to retrieve all avaible ticks from snapshots.
        /// NOTE: if it is null, then use latest tick for list attribute querying.</param>
        /// <param name="tick_length">Number of ticks to query.</param>
        /// <param name="node_indices">Indices of node instance to query, leave it as null to retrieve all node instance from snapshots.
        /// NOTE: it cannot be null if qury for list attribute</param>
        /// <param name="node_length">Number of node instance to query.</param>
        /// <param name="attributes">Attribute type list to query, cannot be null.
        /// NOTE: if first attribute if a list attribute, then there will be a list querying, means only support 1 tick, 1 node, 1 attribute querying.
        /// </param>
        /// <param name="attr_length">Target node type.</param>
        /// <returns>Result shape for input query parameters.</returns>
        SnapshotQueryResultShape prepare(NODE_TYPE node_type, int ticks[], UINT tick_length,
          NODE_INDEX node_indices[], UINT node_length, ATTR_TYPE attributes[], UINT attr_length);

        /// <summary>
        /// Qeury with parameters from prepare function.
        /// </summary>
        /// <param name="result">Pointer to list to hold result value. NOTE: query function will leave the default value for padding.</param>
        void query(QUERY_FLOAT* result);

        /// <summary>
        /// Cancel current querying, this will clear the parameters from last prepare calling.
        /// </summary>
        void cancel_query() noexcept;
      };

      /// <summary>
      /// Tick not supported, like negative tick
      /// </summary>
      struct SnapshotTickError : public exception
      {
        const char* what() const noexcept override;
      };

      /// <summary>
      /// Snapshot list max size is 0
      /// </summary>
      struct SnapshotSizeError : public exception
      {
        const char* what() const noexcept override;
      };

      /// <summary>
      /// Query without call prepare function
      /// </summary>
      struct SnapshotQueryNotPreparedError : public exception
      {
        const char* what() const noexcept override;
      };

      /// <summary>
      /// Attribute not exist when querying
      /// </summary>
      struct SnapshotQueryNoAttributesError : public exception
      {
        const char* what() const noexcept override;
      };

      /// <summary>
      /// Frame not set before operations
      /// </summary>
      struct SnapshotInvalidFrameStateError : public exception
      {
        const char* what() const noexcept override;
      };

      /// <summary>
      /// Array pointer is nullptr
      /// </summary>
      struct SnapshotQueryResultPtrNullError : public exception
      {
        const char* what() const noexcept override;
      };

      struct SnapshotQueryInvalidTickError : public exception
      {
        const char* what() const noexcept override;
      };

      struct SnapshotQueryNoSnapshotsError : public exception
      {
        const char* what() const noexcept override;
      };

      struct SnapshotListQueryNoNodeIndexError : public exception
      {
        const char* what() const noexcept override;
      };
    }
  }
}


#endif // !_MARO_BACKENDS_RAW_SNAPSHOTLIST_
