#include "frame.h"


namespace maro
{
  namespace backends
  {
    namespace raw
    {

      Attribute& maro::backends::raw::Frame::operator()(NODE_INDEX node_index, IDENTIFIER attr_id, SLOT_INDEX slot_index)
      {
        ensure_attr_id(attr_id);

        auto& attr = _attributes[attr_id];
        auto& node = _nodes[attr.node_id];

        ensure_node_index(node, node_index);
        ensure_slot_index(attr, slot_index);

        return _attr_store(attr.node_id, node_index, attr_id, slot_index);
      }

      IDENTIFIER maro::backends::raw::Frame::new_node(string name, USHORT number)
      {
        auto node = FrameNode();
        node.name = name;
        node.number = number;
        node.id = IDENTIFIER(_nodes.size());

        _nodes.push_back(node);

        _node_2_attrs[node.id] = vector<IDENTIFIER>();

        return node.id;
      }

      IDENTIFIER maro::backends::raw::Frame::new_attr(IDENTIFIER node_id, string name, AttrDataType type, SLOT_INDEX slots)
      {
        ensure_node_id(node_id);

        auto& node = _nodes[node_id];

        auto attr = FrameAttribute();
        attr.type = type;
        attr.id = IDENTIFIER(_attributes.size());
        attr.name = name;
        attr.slots = slots;
        attr.node_id = node.id;

        _attributes.push_back(attr);

        _node_2_attrs[node.id].push_back(attr.id);

        return attr.id;
      }

      void Frame::setup()
      {
        for (auto iter : _node_2_attrs)
        {
          auto node_id = iter.first;

          auto& node = _nodes[node_id];

          for (auto attr_id : iter.second)
          {
            auto& attr = _attributes[attr_id];

            _attr_store.add_nodes(node_id, 0, node.number, attr_id, attr.slots);
          }
        }
      }

      void maro::backends::raw::Frame::append_nodes(IDENTIFIER node_id, NODE_INDEX number)
      {
        // to add additional node, the id must exist
        ensure_node_id(node_id);

        auto& node = _nodes[node_id];

        auto attrs_iter = _node_2_attrs.find(node_id);

        // update node number
        node.number += number;

        // add attributes for new nodes
        for (auto attr_id : attrs_iter->second)
        {
          auto attr = _attributes[attr_id];

          // NOTE:
          // attribute store expect that the number is the total number, it will ignore exist node indices
          _attr_store.add_nodes(node_id, 0, node.number, attr_id, attr.slots);
        }
      }

      void maro::backends::raw::Frame::remove_node(IDENTIFIER node_id, NODE_INDEX index)
      {
        ensure_node_id(node_id);

        auto& node = _nodes[node_id];

        ensure_node_index(node, index);

        // remove attributes of this node
        for (auto attr_id : _node_2_attrs.find(node_id)->second)
        {
          auto& attr = _attributes[attr_id];

          _attr_store.remove_node(node_id, index, attr_id, attr.slots);
        }
      }

      void Frame::resume_node(IDENTIFIER node_id, NODE_INDEX index)
      {
        ensure_node_id(node_id);

        auto& node = _nodes[node_id];

        ensure_node_index(node, index);

        for (auto attr_id : _node_2_attrs.find(node_id)->second)
        {
          auto& attr = _attributes[attr_id];

          _attr_store.add_nodes(node_id, index, index + 1, attr_id, attr.slots);
        }
      }

      void maro::backends::raw::Frame::set_attr_slot(IDENTIFIER attr_id, SLOT_INDEX slots)
      {
        // NOTE:
        // set attributes slots will extend or narrow down from the end!
        ensure_attr_id(attr_id);

        auto& attr = _attributes[attr_id];
        auto& node = _nodes[attr.node_id];

        if (slots > attr.slots)
        {
          // extend
          _attr_store.add_nodes(node.id, 0, node.number, attr_id, slots);
        }
        else if (slots < attr.slots)
        {
          // narrow down
          _attr_store.remove_attr_slots(node.id, node.number, attr_id, attr.slots - slots + 1, attr.slots);
        }

        attr.slots = slots;
      }

      inline void Frame::ensure_node_id(IDENTIFIER node_id)
      {
        if (node_id >= _nodes.size())
        {
          throw BadNodeIdentifier();
        }
      }

      inline void Frame::ensure_attr_id(IDENTIFIER attr_id)
      {
        if (attr_id >= _attributes.size())
        {
          throw BadAttributeIdentifier();
        }
      }

      inline void Frame::ensure_node_index(FrameNode& node, NODE_INDEX node_index)
      {
        if (node_index >= node.number)
        {
          throw BadNodeIndex();
        }
      }

      inline void Frame::ensure_slot_index(FrameAttribute& attr, SLOT_INDEX slot_index)
      {
        if (slot_index >= attr.slots)
        {
          throw BadAttributeSlotIndex();
        }
      }

    }
  }
}
