#ifndef _MARO_BACKEND_RAW_FRAME
#define _MARO_BACKEND_RAW_FRAME

#include <map>
#include <vector>

#include "common.h"
#include "attribute.h"
#include "attributestore.h"

using namespace std;

namespace maro
{
  namespace backends
  {
    namespace raw
    {
      class BadNodeIdentifier : public exception
      {
      };

      class BadAttributeIdentifier : public exception
      {
      };

      class BadNodeIndex : public exception
      {
      };

      class BadAttributeSlotIndex : public exception
      {
      };

      struct FrameNode
      {
        IDENTIFIER id;
        // would be changed
        USHORT number;
        USHORT origin_number; // used to reset
        string name;
      };

      struct FrameAttribute
      {
        // slots would be changed
        AttrDataType type;
        SLOT_INDEX slots;
        SLOT_INDEX origin_slots; // used to reset
        IDENTIFIER id;
        IDENTIFIER node_id;
        string name;
      };

      /// <summary>
      /// Used to hold data and node information for current frame
      /// </summary>
      class Frame
      {
        friend class SnapshotList;

        // storage of attributes
        AttributeStore _attr_store;

        // attribute information
        vector<FrameAttribute> _attributes;

        // node information
        vector<FrameNode> _nodes;

        // mapping used to get attributes by node
        map<IDENTIFIER, vector<IDENTIFIER>> _node_2_attrs;

      public:
        /// <summary>
        /// Get attribute at specified slot
        /// </summary>
        /// <param name="node_id">Id of node</param>
        /// <param name="node_index">Index of node</param>
        /// <param name="attr_id">Id of attribue</param>
        /// <param name="slot_index">Index of slot</param>
        /// <returns></returns>
        Attribute &operator()(NODE_INDEX node_index, IDENTIFIER attr_id, SLOT_INDEX slot_index);

        /// <summary>
        /// Add a new node type
        /// </summary>
        /// <param name="name"></param>
        /// <param name="number"></param>
        IDENTIFIER new_node(string name, USHORT number);

        /// <summary>
        /// Add a new attribute type for specified node
        /// </summary>
        /// <param name="node_id"></param>
        /// <param name="name"></param>
        /// <param name="slots"></param>
        IDENTIFIER new_attr(IDENTIFIER node_id, string name, AttrDataType type, SLOT_INDEX slots = 1);

        void setup();

        /// <summary>
        /// Add additional node for specified id
        /// </summary>
        /// <param name="node_id">Id of node to add</param>
        /// <param name="number">Number of nodes to add</param>
        void append_nodes(IDENTIFIER node_id, NODE_INDEX number);

        /// <summary>
        /// Remove specified index node, this will update the node number, but other's index will not change
        /// </summary>
        void remove_node(IDENTIFIER node_id, NODE_INDEX index);

        void resume_node(IDENTIFIER node_id, NODE_INDEX index);

        /// <summary>
        /// Set slot number of specified attribute
        /// </summary>
        /// <param name="attr_id"></param>
        /// <param name="slots"></param>
        void set_attr_slot(IDENTIFIER attr_id, SLOT_INDEX slots);

        USHORT get_node_number(IDENTIFIER node_id);

        USHORT get_slots_number(IDENTIFIER attr_id);

        void reset();

      private:
        inline void ensure_node_id(IDENTIFIER node_id);
        inline void ensure_attr_id(IDENTIFIER attr_id);
        inline void ensure_node_index(FrameNode &node, NODE_INDEX node_index);
        inline void ensure_slot_index(FrameAttribute &attr, SLOT_INDEX slot_index);
      };
    } // namespace raw
  }   // namespace backends
} // namespace maro

#endif
