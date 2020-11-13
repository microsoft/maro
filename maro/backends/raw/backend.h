#ifndef _MARO_BACKENDS_RAW_BACKEND
#define _MARO_BACKENDS_RAW_BACKEND

#include <string>

#include "common.h"
#include "frame.h"
#include "snapshotlist.h"

using namespace std;

namespace maro
{
  namespace backends
  {
    namespace raw
    {
      /// <summary>
      /// Operations before setting up
      /// </summary>
      class InvalidSetupState : public exception
      {
      };

      /// <summary>
      /// Wrapper for out-side
      /// </summary>
      class Backend
      {
        // current frame
        Frame _frame;

        // snapshot list
        SnapshotList _snapshot;

        // if setup called
        bool _is_setup{false};

        // is snapshot enabled
        bool _is_snapshot_enabled{false};

      public:
        /// <summary>
        /// Add a new node type
        /// </summary>
        /// <param name="node_name">Name of the new node</param>
        /// <param name="node_num">Number of the new node</param>
        /// <returns>Id of new node, used for furthur operations</returns>
        IDENTIFIER add_node(string node_name, NODE_INDEX node_num);

        /// <summary>
        /// Add attribute for specified node
        /// </summary>
        /// <param name="node_id">Id of node to add</param>
        /// <param name="attr_name">Name of new attribute</param>
        /// <param name="attr_type">Data type of new attributes</param>
        /// <param name="slot_number">Number of this attributes</param>
        /// <returns>Id of new attribute, used for furthure operations</returns>
        IDENTIFIER add_attr(IDENTIFIER node_id, string attr_name, AttrDataType attr_type, SLOT_INDEX slot_number);

        /// <summary>
        /// Setup node add attribute, after setting up, cannot add a new node type and new attribute type.
        ///
        /// But can dynamically add existing node type
        /// </summary>
        /// <param name="enable_snapshot">If enable snapshot</param>
        /// <param name="snapshot_number">Max number of snapshots</param>
        void setup(bool enable_snapshot, USHORT snapshot_number);

        // getters
        ATTR_BYTE get_byte(IDENTIFIER att_id, NODE_INDEX node_index, SLOT_INDEX slot_index);
        ATTR_SHORT get_short(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index);
        ATTR_INT get_int(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index);
        ATTR_LONG get_long(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index);
        ATTR_FLOAT get_float(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index);
        ATTR_DOUBLE get_double(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index);

        // setter
        template <typename T>
        void set_attr_value(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index, T value);

        /// <summary>
        /// Take snapshot for current frame with a specified tick as key for later querying.
        /// </summary>
        /// <param name="tick">Tick of current frame</param>
        void take_snapshot(INT tick);

        /// <summary>
        /// Reset current frame, this will set all values to 0, all nodes and attributs to original number
        /// </summary>
        void reset_frame();

        /// <summary>
        /// Reset snapshots, this will remove all snapshots.
        /// </summary>
        void reset_snapshots();

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
        void query(QUERING_FLOAT *result, SnapshotResultShape shape);

        /// <summary>
        /// Get max number of snapshots
        /// </summary>
        /// <returns>Max number of snapshots</returns>
        USHORT get_max_snapshot_number();

        /// <summary>
        /// Get number of ticks in snapshot list
        /// </summary>
        /// <returns>Number of ticks</returns>
        USHORT get_valid_tick_number();

        /// <summary>
        /// Get tick list of snapshot list
        /// </summary>
        /// <param name="result">Array to hold ticks</param>
        void get_ticks(INT *result);

        /// <summary>
        /// Dump current frame to specified folder, node will splited into different file (csv)
        /// </summary>
        /// <param name="path">Path to place the dump file.</param>
        void dump_current_frame(string path);

        /// <summary>
        /// Dump current snapshot list to specified folder, node will be splitted into different files (csv)
        /// </summary>
        /// <param name="path">Path to place the dump file</param>
        void dump_snapshots(string path);

        /// <summary>
        /// Append nodes to exist node type
        /// </summary>
        /// <param name="node_id">Id of the node</param>
        /// <param name="number">Number to append</param>
        void append_node(IDENTIFIER node_id, NODE_INDEX number);

        /// <summary>
        /// Delete a node, this will node cause the index changing.
        /// </summary>
        /// <param name="node_id">Id of node</param>
        /// <param name="node_index">Index to delete</param>
        void delete_node(IDENTIFIER node_id, NODE_INDEX node_index);

        /// <summary>
        /// Resume a deleted node
        ///
        /// NOTE: this function cannot get previous value back!
        /// </summary>
        /// <param name="node_id">Id of node</param>
        /// <param name="node_index">Index of node to resume</param>
        void resume_node(IDENTIFIER node_id, NODE_INDEX node_index);

        /// <summary>
        /// Set slot number of specified attribute
        /// </summary>
        /// <param name="attr_id">Id of attribute</param>
        /// <param name="slots">Slot number to set</param>
        void set_attribute_slot(IDENTIFIER attr_id, SLOT_INDEX slots);

        /// <summary>
        /// Get node number of specified node
        /// </summary>
        /// <param name="node_id">Id of node</param>
        /// <returns>Number of node</returns>
        USHORT get_node_number(IDENTIFIER node_id);

        /// <summary>
        /// Get slot number of specified attribute
        /// </summary>
        /// <param name="attr_id">Id of attribute</param>
        /// <returns>Number of slot</returns>
        USHORT get_slots_number(IDENTIFIER attr_id);

      private:
        /// <summary>
        /// Ensure the setup state is same as expected, or cause exception
        /// </summary>
        /// <param name="expect">Expected state</param>
        inline void ensure_setup_state(bool expect);
      };
    } // namespace raw
  }   // namespace backends
} // namespace maro

#endif
