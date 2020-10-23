#ifndef _MARO_BACKEND_RAW_BACKEND
#define _MARO_BACKEND_RAW_BACKEND

/*
Backend used to group attributes into nodes, as Node is a concept here.

Basically backend contains 2 part:

1. current frame:

Used to hold latest attribute information, writable for outside.

current frame is consist with 2 parts:

1). changale area

2). un-changable area


2. snapshot list (only contains attributes that will be in snapshot):

Use to save list of snapshots for current frame (changable attributes), this would be readly only for outside



POSSIABLE STRUCTURES:

1. fixed


|--------------------- current frame ------------------------------|
|              changable area               |  un-changable area   |
|    node 1            |         node 2     |     node 1           |
|-------------------------------- ----------|----------------------|
| a11 | a12 | a2 | a3 | a4 | a5 | a61 | a62 | a7 | a81 | a82 | a83 |

|----------------- snapshot 1 --------------|
|               changable area              |
|    node 1            |         node 2     |
|-------------------------------------------|
| a11 | a12 | a2 | a3 | a4 | a5 | a61 | a62 |

|----------------- snapshot N --------------|
|               changable area              |
|    node 1            |         node 2     |
|-------------------------------------------|
| a11 | a12 | a2 | a3 | a4 | a5 | a61 | a62 |




2. complex

memory layout (4 parts, we may use memmap with 4 files to hold this):

|     indices memory      |          frame memory                     |
|-------------------------|-------------------------------------------|
|     indices memory      |          snapshot memory                  |


We can also have a global index table for fast accessing.


|   tick    |    node_id    | node_index    |  attr_id   | slot_index  |


Structure:

|------------------------------- current frame ----------------------------------------------|
|-------------------------|----------- changable area --------------- |  un-changable area   |
|-------------------------|      node 1         |         node 2      |     node 1           |
|      node indices       |-------------------------------------------|----------------------|
| n1-indices | n2-indices | a11 | a12 | a2 | a3 | a4 | a5 | a61 | a62 | a7 | a81 | a82 | a83 |


|------------------------------- snapshot 1 --------------------------|
|-------------------------|------------------ changable area ---------|
|-------------------------|    node 1            |         node 2     |
|  node indices           |-------------------------------- ----------|
| n1-indices | n2-indices | a11 | a12 | a2 | a3 | a4 | a5 | a61 | a62 |


|--------------------------------snapshot N------------------------------------------------|
|-------------------------|--------------------- changable area ---------------------------|
|-------------------------|      node 1         |         node 2      |         node 3     |
|-----node indices--------|----------------------------------------------------------------|
| n1-indices | n2-indices | a11 | a12 | a2 | a3 | a4 | a5 | a61 | a62 | a7 | a8 | a9 | a10 |


To support dynamic nodes (add/remove), we have 2 method:

1. No in-memory snapshot over-write

 do not support snapshot overwrite, just keep all snapshot in mem-mapping file. we can just expend the file to support more data


2. with in memory snapshot over-write
 we need another mapping that used to index every part of snapshot, as the length will be different for snapshots.

 When over-writing:
 1). if existing snapshot is larger than current one, then we just save current frame in current snapshot, and leave the additional space there
 2). if existing snapshot is shorter than current one, then we have 2 way to due with:
    1)). split current frame into 2 parts, 1st one's length same as over-writing snapshot, 2nd append to the end (after allocate new memory)
    2)). just allocate a large enough meomry to hold this, make over-writing snapshot as avaialble


all above methods need a table to track avaiable and existing snapshot, but may be the content is different
|                      snapshot mapping table                             |





*/

#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>

#include "common.h"
#include "attribute.h"

using namespace std;

namespace maro
{
  namespace backends
  {
    namespace raw
    {
      /// <summary>
      /// Attribute accessing before backend setup
      /// </summary>
      class BadSetupState : public exception
      {
      };


      /// <summary>
      /// Invalid node index to access
      /// </summary>
      class BadNodeIndex : public exception
      {
      };


      /// <summary>
      /// Invalid slot index to access
      /// </summary>
      class BadSlotIndex : public exception
      {
      };


      /// <summary>
      /// Invalid node id to access
      /// </summary>
      class BadNodeIdentifier : public exception
      {
      };


      /// <summary>
      /// Invalid attribute id to access
      /// </summary>
      class BadAttributeIdentifier : public exception
      {

      };


      /// <summary>
      /// Reach the max id of attribute (not slot number)
      /// </summary>
      class MaxAttributeTypeNumberError : public exception
      {
      };


      /// <summary>
      /// Reach the max id of node
      /// </summary>
      class MaxNodeTypeNumberError : public exception
      {
      };


      /// <summary>
      /// Invalid tick to take snapshot, it must be greater than 0 for now
      /// </summary>
      class InvalidSnapshotTick : public exception
      {};

      class MaxSnapshotError: public exception
      {};


      /// <summary>
      /// Backend used to hold node and releated attributes, providing accessing interface 
      /// </summary>
      class Backend
      {
        /// <summary>
        /// Internal structure to hold node information.
        /// </summary>
        struct NodeInfo
        {
          NODE_INDEX number{ 1 };

          // offset in 1 frame
          UINT offset;

          // per node
          ULONG attr_number;

          IDENTIFIER id;

          string name;
        };

        /// <summary>
        /// Internal structure to hold attribute information.
        /// </summary>
        struct AttrInfo
        {
          AttrDataType data_type;

          // offset in node
          UINT offset;

          NODE_INDEX slots{ 1 };

          IDENTIFIER node_id;

          IDENTIFIER id;

          string name;
        };

        // node register information
        vector<NodeInfo> _nodes;

        // attribute register information
        vector<AttrInfo> _attrs;

        // used to hold all the attributes
        vector<Attribute> _data;

        // length of one frame, this is fixed after setup, used for fast calculation
        size_t _frame_length{ 0 };

        // is backend already setup
        bool _is_setup{ false };

        bool _is_snapshot_enabled{ false };

        // capacity number of snapshots to keep
        USHORT _snapshot_number{ 0 };

        // mapping for snapshot from tick to internal index
        unordered_map<INT, INT> _ss_tick2index_map;

        // mapping for snapshot from internal index to tick
        unordered_map<INT, INT> _ss_index2tick_map;

        // 0 is current frame
        INT _cur_snapshot_index{ 1 };

      public:
        Backend();

        /// <summary>
        /// Add a new node in backend
        /// </summary>
        /// <param name="node_name">Name of new node</param>
        /// <returns>Id of new node, NOTE: this id is different with index, it is used to identify a node</returns>
        IDENTIFIER add_node(string node_name) noexcept;

        /// <summary>
        /// Add a new attribute to specified node
        /// </summary>
        /// <param name="node_id">Id of node which new the attribute belongs to</param>
        /// <param name="attr_name">Name of new attribute</param>
        /// <param name="attr_type">Data type of this attribute</param>
        /// <param name="slot_number">Number of slots of this attribute, default is 1</param>
        /// <returns>Id of the new attribute</returns>
        IDENTIFIER add_attr(IDENTIFIER node_id, string attr_name, AttrDataType attr_type, SLOT_INDEX slot_number = 1);

        /// <summary>
        /// Set value for specified attribute's slot
        /// </summary>
        /// <typeparam name="T">Supported attribute data type</typeparam>
        /// <param name="attr_id">Id of attribute</param>
        /// <param name="node_index">Index of node the attribute belongs to</param>
        /// <param name="slot_index">Index of slot to set</param>
        /// <param name="value">Value to set</param>
        template <typename T>
        void set_attr_value(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index, T value);

        // Attribute getters
        ATTR_BYTE get_byte(IDENTIFIER att_id, NODE_INDEX node_index, SLOT_INDEX slot_index);
        ATTR_SHORT get_short(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index);
        ATTR_INT get_int(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index);
        ATTR_LONG get_long(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index);
        ATTR_FLOAT get_float(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index);
        ATTR_DOUBLE get_double(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index);

        /// <summary>
        /// Set number of nodes in this backend
        /// </summary>
        /// <param name="node_id">Id of node to set</param>
        /// <param name="number">Number to set</param>
        void set_node_number(IDENTIFIER node_id, NODE_INDEX number);

        /// <summary>
        /// Setup current backend。
        ///
        /// After seting up, add_attr and add_node will not work, and attribute's getter/setter can work at this point
        /// </summary>
        /// <param name="enable_snapshot">If backend should enable snapshot</param>
        /// <param name="snapshot_number">Number of snapshots (in-memory) should be keep, this means old one will be over-write if reach the capacity</param>
        void setup(bool enable_snapshot, USHORT snapshot_number);

        /// <summary>
        /// Reset backend frame to initial
        /// </summary>
        void reset_frame();

        /// <summary>
        /// Reset backend snapshots to initial
        /// </summary>
        void reset_snapshots();

        /*****************************************************/
        // Snapshot functions

        /*
        We keep snapshot related function here as we will use one vector to hold both current frame and snapshots
        */

        /// <summary>
        /// Take a snapshot for current frame
        /// </summary>
        /// <param name="tick">Key of this snapshot</param>
        void take_snapshot(INT tick);

        /// <summary>
        /// Get length of 1 tick querying
        /// </summary>
        /// <param name="node_id">Id of target node</param>
        /// <param name="node_indices">Array of node index to query</param>
        /// <param name="node_length">Length of node index array</param>
        /// <param name="attributes">Id array of quering attribute</param>
        /// <param name="attr_length">Length of attribute array</param>
        /// <returns>Length of one tick for this query</returns>
        UINT query_one_tick_length(IDENTIFIER node_id, const NODE_INDEX node_indices[], UINT node_length, const IDENTIFIER attributes[], UINT attr_length);

        /// <summary>
        /// Query in snapshots with conditions
        /// </summary>
        /// <param name="result">Float pointer that used to hold result, it should be big enough</param>
        /// <param name="node_id">Id of target node</param>
        /// <param name="ticks">Ticks to query</param>
        /// <param name="ticks_length">Length of ticks length</param>
        /// <param name="node_indices">Array of node index to query</param>
        /// <param name="node_length">Length of node index array</param>
        /// <param name="attributes">Id array of quering attribute</param>
        /// <param name="attr_length">Length of attribute array</param>
        void query(ATTR_FLOAT* result, IDENTIFIER node_id, const INT ticks[], UINT ticks_length, const NODE_INDEX node_indices[], UINT node_length, const IDENTIFIER attributes[], UINT attr_length);

        /// <summary>
        /// Get node number for specified node
        /// </summary>
        /// <param name="node_id">Id of node to get</param>
        /// <returns>Number of this node</returns>
        NODE_INDEX get_node_number(IDENTIFIER node_id);

        /// <summary>
        /// Get max number of snapshots
        /// </summary>
        /// <returns>Number of max snapshots</returns>
        USHORT get_max_snapshot_number();

        /// <summary>
        /// How many ticks in snapshot
        /// </summary>
        /// <returns>Number of ticks in snapshots</returns>
        USHORT get_valid_tick_number();

      private:

        // Used to ensure setup state for further operations
        inline void ensure_setup_state(bool expected_state = false);

        /// <summary>
        /// Ensure that node index is valid.
        /// </summary>
        /// <param name="cur">Current node index</param>
        /// <param name="node">Target node information</param>
        inline void ensure_node_index(NODE_INDEX cur, NodeInfo& node);

        /// <summary>
        /// Ensure that slot index is valid.
        /// </summary>
        /// <param name="cur">Current slot index</param>
        /// <param name="attr">Target attribute information</param>
        inline void ensure_slot_index(SLOT_INDEX cur, AttrInfo& attr);

        /// <summary>
        /// Ensure that node id is valid
        /// </summary>
        /// <param name="id">Node id</param>
        inline void ensure_node_id(IDENTIFIER id);

        /// <summary>
        /// Ensure attribute id is valid
        /// </summary>
        /// <param name="id">Attribute id</param>
        inline void ensure_attr_id(IDENTIFIER id);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="node"></param>
        /// <param name="node_index"></param>
        /// <param name="attr"></param>
        /// <param name="slot_index"></param>
        /// <returns></returns>
        inline size_t calc_attr_index(UINT frame_index, NodeInfo& node, NODE_INDEX node_index, AttrInfo& attr, SLOT_INDEX slot_index);

      };
    } // namespace raw
  }     // namespace backends
} // namespace maro

#endif
