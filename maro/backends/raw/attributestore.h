#ifndef _MARO_BACKENDS_RAW_ATTRIBUTESTORE
#define _MARO_BACKENDS_RAW_ATTRIBUTESTORE

#include <vector>
#include <unordered_map>
#include <iterator>

#include "common.h"
#include "attribute.h"
#include "bitset.h"


namespace maro
{
  namespace backends
  {
    namespace raw
    {
      class BadAttributeIndexing : public exception
      {
      };

      inline ULONG attr_index_key(IDENTIFIER node_id, NODE_INDEX node_index, IDENTIFIER attr_id, SLOT_INDEX slot_index);

      const USHORT LENGTH_PER_PART = sizeof(USHORT) * BITS_PER_BYTE;

      /*
      NOTE:
        1. removing will not change last index
        2. adding will increase last index
        3. arrange will update last index when filling the empty slots
      */

      class AttributeStore
      {
        // attribute mapping: [node_id, node_index, attr_id, slot_index] -> attribute index
        unordered_map<ULONG, size_t> _mapping;

        // attribute index -> [node_id, node_index, attr_id, slot_index]
        unordered_map<size_t, ULONG> _i2kmaping;

        vector<Attribute> _attributes;

        //Bitset _slot_mask;
        Bitset _slot_masks;

        // arrange if dirty
        bool _is_dirty{ false };

        // index of last attribute
        size_t _last_index{ 0ULL };


      public:

        /// <summary>
        /// Setup attributes store.
        /// </summary>
        /// <param name="size">Initial size of store, this would be 64 times, it will be expend to times of 64 if not</param>
        void setup(size_t size);

        /// <summary>
        /// Arrange attribute store to avoid empty slots in the middle
        /// </summary>
        void arrange();

        /// <summary>
        /// Attribute getter to support get attribue like: auto& attr = store(node_id, node_index, attr_id, slot_index)
        /// </summary>
        /// <param name="node_id">IDENTIFIER of node</param>
        /// <param name="node_index">Index of node</param>
        /// <param name="attr_id">IDENTIFIER of attribute</param>
        /// <param name="slot_index">Slot of attribute</param>
        /// <returns>Attribute at specified place</returns>
        Attribute& operator()(IDENTIFIER node_id, NODE_INDEX node_index, IDENTIFIER attr_id, SLOT_INDEX slot_index = 0);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="node_id"></param>
        /// <param name="node_start_index"></param>
        /// <param name="node_num"></param>
        /// <param name="attr_id"></param>
        /// <param name="slot_num"></param>
        void add_nodes(IDENTIFIER node_id, NODE_INDEX node_start_index, NODE_INDEX stop, IDENTIFIER attr_id, SLOT_INDEX slot_num);

        /// <summary>
        /// Remove specified node
        /// </summary>
        /// <param name="node_id"></param>
        /// <param name="node_index"></param>
        /// <param name="attr_id"></param>
        /// <param name="slot_num"></param>
        void remove_node(IDENTIFIER node_id, NODE_INDEX node_index, IDENTIFIER attr_id, SLOT_INDEX slot_num);

        /// <summary>
        /// Remove specified range of attribute slots
        /// </summary>
        /// <param name="node_id"></param>
        /// <param name="node_num"></param>
        /// <param name="attr_id"></param>
        /// <param name="from"></param>
        /// <param name="stop"></param>
        void remove_attr_slots(IDENTIFIER node_id, NODE_INDEX node_num, IDENTIFIER attr_id, SLOT_INDEX from, SLOT_INDEX stop);

        /// <summary>
        /// Copy valid data to target address
        /// </summary>
        void copy_to(Attribute* p, unordered_map<ULONG, size_t> &map);

        size_t size();

        void reset();

#ifdef _DEBUG
        size_t capacity();
        size_t last_index();
#endif

      private:
        void update_last_index();
      };
    }
  }
}

#endif
