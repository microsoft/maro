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
      struct FrameNode
      {
        IDENTIFIER id;
        // would be changed
        USHORT number;
        string name;
      };

      struct FrameAttribute
      {
        // slots would be changed
        SLOT_INDEX slots;
        IDENTIFIER id;
        IDENTIFIER node_id;
        string name;
      };

      /// <summary>
      /// Used to hold data and node information for current frame
      /// </summary>
      class Frame
      {
        // storage of attributes
        AttributeStore _attr_store;

        // attribute information
        vector<FrameAttribute> _attributes;

        // node information
        vector<FrameNode> _nodes;

      public:
        /// <summary>
        /// Get attribute at specified slot
        /// </summary>
        /// <param name="node_id">Id of node</param>
        /// <param name="node_index">Index of node</param>
        /// <param name="attr_id">Id of attribue</param>
        /// <param name="slot_index">Index of slot</param>
        /// <returns></returns>
        Attribute& operator()(NODE_INDEX node_index, IDENTIFIER attr_id, SLOT_INDEX slot_index);


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
        IDENTIFIER new_attr(IDENTIFIER node_id, string name, SLOT_INDEX slots);


        /// <summary>
        /// Add additional node for specified id
        /// </summary>
        /// <param name="node_id">Id of node to add</param>
        /// <param name="number">Number of nodes to add</param>
        void add_node(IDENTIFIER node_id, NODE_INDEX number);

        /// <summary>
        /// Remove specified index node, this will update the node number, but other's index will not change
        /// </summary>
        void remove_node(IDENTIFIER node_id, NODE_INDEX index);


        /// <summary>
        /// Set slot number of specified attribute
        /// </summary>
        /// <param name="attr_id"></param>
        /// <param name="slots"></param>
        void set_attr_slot(IDENTIFIER attr_id, SLOT_INDEX slots);


        void copy_to(void* p);
      };
    }
  }
}



#endif
