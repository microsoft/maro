#include "frame.h"

namespace maro
{
  namespace backends
  {
    namespace raw
    {

      Attribute &maro::backends::raw::Frame::operator()(NODE_INDEX node_index, IDENTIFIER attr_id, SLOT_INDEX slot_index)
      {
        ensure_attr_id(attr_id);

        auto &attr = _attributes[attr_id];
        auto &node = _nodes[attr.node_id];

        ensure_node_index(node, node_index);
        ensure_slot_index(attr, slot_index);

        return _attr_store(attr.node_id, node_index, attr_id, slot_index);
      }

      IDENTIFIER maro::backends::raw::Frame::new_node(string name, USHORT number)
      {
        _nodes.emplace_back();

        auto &node = _nodes.back();
        node.name = name;
        node.number = number;
        node.origin_number = number;
        node.id = IDENTIFIER(_nodes.size() - 1);

        _node_2_attrs[node.id] = vector<IDENTIFIER>();

        return node.id;
      }

      IDENTIFIER maro::backends::raw::Frame::new_attr(IDENTIFIER node_id, string name, AttrDataType type, SLOT_INDEX slots)
      {
        ensure_node_id(node_id);

        auto &node = _nodes[node_id];

        _attributes.emplace_back();

        auto& attr = _attributes.back();
        attr.type = type;
        attr.id = IDENTIFIER(_attributes.size() - 1);
        attr.name = name;
        attr.slots = slots;
        attr.origin_slots = slots;
        attr.max_slots = slots;
        attr.node_id = node.id;

        auto &node_attrs_list = _node_2_attrs[node.id];

        node_attrs_list.push_back(attr.id);

        return attr.id;
      }

      void Frame::setup()
      {
        for (auto iter : _node_2_attrs)
        {
          auto node_id = iter.first;

          auto &node = _nodes[node_id];

          for (auto attr_id : iter.second)
          {
            auto &attr = _attributes[attr_id];

            _attr_store.add_nodes(node_id, 0, node.number, attr_id, attr.slots);
          }
        }
      }

      void maro::backends::raw::Frame::append_nodes(IDENTIFIER node_id, NODE_INDEX number)
      {
        // to add additional node, the id must exist
        ensure_node_id(node_id);

        auto &node = _nodes[node_id];

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

        auto &node = _nodes[node_id];

        ensure_node_index(node, index);

        // remove attributes of this node
        for (auto attr_id : _node_2_attrs.find(node_id)->second)
        {
          auto &attr = _attributes[attr_id];

          _attr_store.remove_node(node_id, index, attr_id, attr.slots);
        }
      }

      void Frame::resume_node(IDENTIFIER node_id, NODE_INDEX index)
      {
        ensure_node_id(node_id);

        auto &node = _nodes[node_id];

        ensure_node_index(node, index);

        for (auto attr_id : _node_2_attrs.find(node_id)->second)
        {
          auto &attr = _attributes[attr_id];

          _attr_store.add_nodes(node_id, index, index + 1, attr_id, attr.slots);
        }
      }

      void maro::backends::raw::Frame::set_attr_slot(IDENTIFIER attr_id, SLOT_INDEX slots)
      {
        // NOTE:
        // set attributes slots will extend or narrow down from the end!
        ensure_attr_id(attr_id);

        auto &attr = _attributes[attr_id];
        auto &node = _nodes[attr.node_id];

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
        attr.max_slots = max(slots, attr.max_slots);
      }

      USHORT Frame::get_node_number(IDENTIFIER node_id)
      {
        ensure_node_id(node_id);

        return _nodes[node_id].number;
      }

      USHORT Frame::get_slots_number(IDENTIFIER attr_id)
      {
        ensure_attr_id(attr_id);

        return _attributes[attr_id].slots;
      }

      void Frame::reset()
      {
        _attr_store.reset();

        // reset node and attr info
        for (auto &node : _nodes)
        {
          node.number = node.origin_number;
        }

        for (auto &attr : _attributes)
        {
          attr.slots = attr.origin_slots;
          attr.max_slots = attr.origin_slots;
        }

        // setup again
        setup();
      }

      void Frame::write_attribute(ofstream &file, NODE_INDEX node_index, IDENTIFIER attr_id, SLOT_INDEX slot_index)
      {
        try
        {
          auto &a = operator()(node_index, attr_id, 0);

          file << ATTR_FLOAT(a);
        }
        catch (const BadAttributeIndexing &e)
        {
          file << "nan";
        }
      }

      void Frame::dump(string path)
      {
        // for dump, we will save for each node, named as "node_<node_name>.csv"
        // content of the csv will follow padans' output that list will be wrapped into a "[]",
        for (auto node : _nodes)
        {
          auto output_path = path + "/" + "node_" + node.name + ".csv";

          ofstream file(output_path);

          // write headers
          file << "node_index";

          for (IDENTIFIER attr_id : _node_2_attrs.find(node.id)->second)
          {
            auto attr = _attributes[attr_id];

            file << "," << attr.name;
          }

          // end of headers
          file << "\n";

          // write for each node
          for (NODE_INDEX node_index = 0; node_index < node.number; node_index++)
          {
            // node index
            file << node_index;

            for (IDENTIFIER attr_id : _node_2_attrs.find(node.id)->second)
            {
              auto attr = _attributes[attr_id];

              if (attr.slots == 1)
              {
                file << ",";

                write_attribute(file, node_index, attr_id, 0);
              }
              else
              {
                // start of list
                file << ",\"[";

                for (SLOT_INDEX slot_index = 0; slot_index < attr.slots; slot_index++)
                {
                  write_attribute(file, node_index, attr_id, slot_index);

                  file << ",";
                }

                // end of list
                file << "]\"";
              }
            }

            // end of row
            file << "\n";
          }

          file.close();
        }
      } // namespace raw

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

      inline void Frame::ensure_node_index(FrameNode &node, NODE_INDEX node_index)
      {
        if (node_index >= node.number)
        {
          throw BadNodeIndex();
        }
      }

      inline void Frame::ensure_slot_index(FrameAttribute &attr, SLOT_INDEX slot_index)
      {
        if (slot_index >= attr.slots)
        {
          throw BadAttributeSlotIndex();
        }
      }

    } // namespace raw
  }   // namespace backends
} // namespace maro
