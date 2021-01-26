// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "node.h"

namespace maro
{
  namespace backends
  {
    namespace raw
    {
      inline USHORT extract_attr_index(ATTR_TYPE attr_type)
      {
        return USHORT(attr_type & 0x0000ffff);
      }

      inline size_t compose_attr_offset_in_node(NODE_INDEX node_index, size_t node_size,
        size_t attr_offset, SLOT_INDEX slot)
      {
        return node_index * node_size + attr_offset + slot;
      }

      AttributeDef::AttributeDef(string name, AttrDataType data_type, SLOT_INDEX slot_number,
        size_t offset, bool is_list, bool is_const, ATTR_TYPE attr_type) :
        name(name),
        slot_number(slot_number),
        offset(offset),
        is_list(is_list),
        is_const(is_const),
        data_type(data_type),
        attr_type(attr_type)
      {
      }

      void Node::copy_from(const Node& node, bool is_deep_copy)
      {
        // Copy normal fields.
        _dynamic_size_per_node = node._dynamic_size_per_node;
        _const_size_per_node = node._const_size_per_node;
        _max_node_number = node._max_node_number;
        _alive_node_number = node._alive_node_number;
        _defined_node_number = node._defined_node_number;
        _type = node._type;
        _is_setup = node._is_setup;

        // Ignore name.
        _name = "";

        // Copy dynamic block.
        if (node._dynamic_block.size() > 0)
        {
          // Copy according to max_node number, as memory block may larger than it (after reset).
          auto valid_dynamic_size = node._dynamic_size_per_node * node._max_node_number;

          _dynamic_block.resize(valid_dynamic_size);

          memcpy(&_dynamic_block[0], &node._dynamic_block[0], valid_dynamic_size * sizeof(Attribute));
        }

        // Copy list attributes store.
        if (node._list_store.size() > 0)
        {
          _list_store.resize(node._list_store.size());

          for (size_t i = 0; i < _list_store.size(); i++)
          {
            auto& source_list = node._list_store[i];

            if (source_list.size() > 0)
            {
              auto& target_list = _list_store[i];

              target_list.resize(source_list.size());

              memcpy(&target_list[0], &source_list[0], source_list.size() * sizeof(Attribute));
            }
          }
        }

        // Copy masks.
        _node_instance_masks = node._node_instance_masks;

        // Copy others for deep-copy.
        if (is_deep_copy)
        {
          _name = node.get_name();

          _attribute_definitions = node._attribute_definitions;

          // NOTE: we do not copy const block here, as this operation occur before setting up
          // and there is nothing in const block
        }
      }

      inline void Node::ensure_setup() const
      {
        if (!_is_setup)
        {
          throw OperationsBeforeSetupError();
        }
      }

      inline void Node::ensure_attr_index(USHORT attr_index) const
      {
        if (attr_index >= _attribute_definitions.size())
        {
          throw InvalidAttributeTypeError();
        }
      }

      inline void Node::ensure_node_index(NODE_INDEX node_index) const
      {
        // check is node alive
        if (!_node_instance_masks.get(node_index))
        {
          throw InvalidNodeIndexError();
        }
      }

      Node::Node()
      {
      }

      Node::Node(const Node& node)
      {
        // This function invoked when the node list is increasing its size,
        // then it need to copy nodes to new memory block.
        copy_from(node, true);
      }

      Node& Node::operator=(const Node& node)
      {
        if (this != &node)
        {
          copy_from(node);
        }

        return *this;
      }

      void Node::set_type(NODE_TYPE type) noexcept
      {
        _type = type;
      }

      NODE_TYPE Node::get_type() const noexcept
      {
        return _type;
      }

      void Node::set_name(string name) noexcept
      {
        _name = name;
      }

      string Node::get_name() const noexcept
      {
        return _name;
      }

      void Node::set_defined_number(NODE_INDEX number)
      {
        if (number == 0)
        {
          throw InvalidNodeNumberError();
        }

        _defined_node_number = number;
        _max_node_number = number;
        _alive_node_number = number;
      }

      NODE_INDEX Node::get_defined_number() const noexcept
      {
        return _defined_node_number;
      }

      NODE_INDEX Node::get_max_number() const noexcept
      {
        return _max_node_number;
      }

      const AttributeDef& Node::get_attr_definition(ATTR_TYPE attr_type) const
      {
        USHORT attr_index = extract_attr_index(attr_type);

        ensure_attr_index(attr_index);

        return _attribute_definitions[attr_index];
      }

      bool Node::is_node_alive(NODE_INDEX node_index) const noexcept
      {
        ensure_setup();

        return _node_instance_masks.get(node_index);
      }

      SLOT_INDEX Node::get_slot_number(NODE_INDEX node_index, ATTR_TYPE attr_type) const
      {
        ensure_setup();
        ensure_node_index(node_index);

        auto& attr_def = get_attr_definition(attr_type);

        // If it is a list attribute, we will return actual list size.
        if (attr_def.is_list)
        {
          auto attr_offset = compose_attr_offset_in_node(node_index, _dynamic_size_per_node, attr_def.offset);
          auto& target_attr = _dynamic_block[attr_offset];

          return target_attr.slot_number;
        }

        // Or used pre-defined number.
        return attr_def.slot_number;
      }

      void Node::setup()
      {
        // Ignore is already been setup.
        if (_is_setup)
        {
          return;
        }

        // Initialize dynamic and const block.
        _const_block.resize(_defined_node_number * _const_size_per_node);
        _dynamic_block.resize(_defined_node_number * _dynamic_size_per_node);

        // Prepare bitset for masks.
        _node_instance_masks.resize(_defined_node_number);
        _node_instance_masks.reset(true);

        // Prepare memory for list attributes.
        for (auto& attr_def : _attribute_definitions)
        {
          if (attr_def.is_list)
          {
            // Assign each attribute with the index of actual list.
            for (NODE_INDEX i = 0; i < _defined_node_number; i++)
            {
              auto& target_attr = _dynamic_block[_dynamic_size_per_node * i + attr_def.offset];

              // Save the index of list in list store.
              target_attr = UINT(_list_store.size());

              // Append a new vector for this attribute.
              _list_store.emplace_back();
            }
          }
        }

        _is_setup = true;
      }

      void Node::reset()
      {
        ensure_setup();

        // Reset all node number to pre-defined.
        _max_node_number = _defined_node_number;
        _alive_node_number = _defined_node_number;

        // Clear all attribute to 0.
        memset(&_dynamic_block[0], 0, _dynamic_block.size() * sizeof(Attribute));

        // Clear all list attribute.
        for (auto& list : _list_store)
        {
          list.clear();
        }

        // Reset bitset masks.
        _node_instance_masks.resize(_defined_node_number);
        _node_instance_masks.reset(true);
      }

      void Node::append_nodes(NODE_INDEX node_number)
      {
        ensure_setup();

        if (node_number == 0)
        {
          return;
        }

        _max_node_number += node_number;
        _alive_node_number += node_number;

        // Extend const memory block.
        auto extend_size = _max_node_number * _const_size_per_node;

        if (extend_size > _const_block.size())
        {
          _const_block.resize(extend_size);
        }

        // Extend dynamic memory block.
        extend_size = _max_node_number * _dynamic_size_per_node;

        if (extend_size > _dynamic_block.size())
        {
          _dynamic_block.resize(extend_size);
        }

        // Prepare memory for new list attributes.
        for (auto& attr_def : _attribute_definitions)
        {
          if (attr_def.is_list)
          {
            // Again allocate list.
            for (NODE_INDEX i = 0; i < node_number; i++)
            {
              auto node_index = _max_node_number - node_number + i;
              auto attr_offset = compose_attr_offset_in_node(node_index, _dynamic_size_per_node, attr_def.offset);
              auto& target_attr = _dynamic_block[attr_offset];

              target_attr = UINT(_list_store.size());

              _list_store.emplace_back();
            }
          }
        }

        // Extern masks.
        _node_instance_masks.resize(_max_node_number);

        // Set new node instance as alive.
        for (NODE_INDEX i = 0; i < node_number; i++)
        {
          _node_instance_masks.set(_max_node_number - node_number + i, true);
        }
      }

      void Node::remove_node(NODE_INDEX node_index)
      {
        ensure_setup();
        ensure_node_index(node_index);

        _node_instance_masks.set(node_index, false);
      }

      void Node::resume_node(NODE_INDEX node_index)
      {
        ensure_setup();

        if(node_index < _max_node_number)
        {
          _node_instance_masks.set(node_index, true);
        }
      }

      ATTR_TYPE Node::add_attr(string attr_name, AttrDataType data_type,
        SLOT_INDEX slot_number, bool is_const, bool is_list)
      {
        if (_is_setup)
        {
          throw OperationsAfterSetupError();
        }

        USHORT attr_index = USHORT(_attribute_definitions.size());
        ATTR_TYPE attr_type = UINT(_type) << 16 | attr_index;

        size_t offset = 0;

        // We do not support const list attribute.
        if (is_const && is_list)
        {
          throw InvalidAttributeDescError();
        }

        // List attribute take 1 attribute to hold its list index in list store.
        slot_number = is_list ? 1 : slot_number;

        // Calculate size of each node instance in different memory block.
        if (is_const)
        {
          offset = _const_size_per_node;

          _const_size_per_node += slot_number;
        }
        else
        {
          offset = _dynamic_size_per_node;

          _dynamic_size_per_node += slot_number;
        }

        _attribute_definitions.emplace_back(attr_name, data_type, slot_number, offset, is_list, is_const, attr_type);

        return attr_type;
      }

      Attribute& Node::get_attr(NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX slot_index)
      {
        ensure_setup();
        ensure_node_index(node_index);

        auto& attr_def = get_attr_definition(attr_type);

        if (attr_def.is_list)
        {
          // For list attribute, we need to get its index in list store.
          auto attr_offset = compose_attr_offset_in_node(node_index, _dynamic_size_per_node, attr_def.offset);
          auto& target_attr = _dynamic_block[attr_offset];

          // Slot number of list attribute save in itr attribute.
          if (slot_index >= target_attr.slot_number)
          {
            throw InvalidSlotIndexError();
          }

          const auto list_index = target_attr.get_value<ATTR_UINT>();

          // Then get the actual list reference for furthure operation.
          auto& target_list = _list_store[list_index];

          return target_list[slot_index];
        }

        // Check slot number for normal attributes.
        if (slot_index >= attr_def.slot_number)
        {
          throw InvalidSlotIndexError();
        }

        // Get attribute for const and dynamic attribute.
        vector<Attribute>* target_block = nullptr;
        size_t node_size = 0;

        if (attr_def.is_const)
        {
          target_block = &_const_block;
          node_size = _const_size_per_node;
        }
        else
        {
          target_block = &_dynamic_block;
          node_size = _dynamic_size_per_node;
        }

        auto attr_offset = compose_attr_offset_in_node(node_index, node_size, attr_def.offset, slot_index);

        return (*target_block)[attr_offset];
      }

      inline Attribute& Node::get_list_attribute(NODE_INDEX node_index, ATTR_TYPE attr_type)
      {
        ensure_setup();
        ensure_node_index(node_index);

        auto& attr_def = get_attr_definition(attr_type);

        if (!attr_def.is_list)
        {
          throw OperationsOnNonListAttributeError();
        }

        auto attr_offset = compose_attr_offset_in_node(node_index, _dynamic_size_per_node, attr_def.offset);
        auto& target_attr = _dynamic_block[attr_offset];

        return target_attr;
      }

      inline vector<Attribute>& Node::get_attribute_list(Attribute& attribute)
      {
        const auto& list_index = attribute.get_value<ATTR_UINT>();

        auto& target_list = _list_store[list_index];

        return target_list;
      }

      void Node::clear_list(NODE_INDEX node_index, ATTR_TYPE attr_type)
      {
        auto& target_attr = get_list_attribute(node_index, attr_type);
        auto& target_list = get_attribute_list(target_attr);

        target_list.clear();

        target_attr.slot_number = 0;
      }

      void Node::resize_list(NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX new_size)
      {
        auto& target_attr = get_list_attribute(node_index, attr_type);
        auto& target_list = get_attribute_list(target_attr);

        target_list.resize(new_size);

        target_attr.slot_number = new_size;
      }

      template<typename T>
      void Node::append_to_list(NODE_INDEX node_index, ATTR_TYPE attr_type, T value)
      {
        auto& target_attr = get_list_attribute(node_index, attr_type);
        auto& target_list = get_attribute_list(target_attr);

        target_list.push_back(value);

        target_attr.slot_number++;
      }

#define APPEND_TO_LIST(type) \
  template void Node::append_to_list(NODE_INDEX node_index, ATTR_TYPE attr_type, type value);

      APPEND_TO_LIST(ATTR_CHAR)
      APPEND_TO_LIST(ATTR_UCHAR)
      APPEND_TO_LIST(ATTR_SHORT)
      APPEND_TO_LIST(ATTR_USHORT)
      APPEND_TO_LIST(ATTR_INT)
      APPEND_TO_LIST(ATTR_UINT)
      APPEND_TO_LIST(ATTR_LONG)
      APPEND_TO_LIST(ATTR_ULONG)
      APPEND_TO_LIST(ATTR_FLOAT)
      APPEND_TO_LIST(ATTR_DOUBLE)

      void Node::remove_from_list(NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX slot_index)
      {
        auto& target_attr = get_list_attribute(node_index, attr_type);
        auto& target_list = get_attribute_list(target_attr);

        if(slot_index >= target_list.size())
        {
          throw InvalidSlotIndexError();
        }

        target_list.erase(target_list.begin() + slot_index);

        target_attr.slot_number--;
      }

      template<typename T>
      void Node::insert_to_list(NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX slot_index, T value)
      {
        auto& target_attr = get_list_attribute(node_index, attr_type);
        auto& target_list = get_attribute_list(target_attr);

        // NOTE: the insert index can same as size, then it is the last one.
        if(slot_index > target_list.size())
        {
          throw InvalidSlotIndexError();
        }

        // Check if reach the max slot number
        if(target_list.size() >= MAX_SLOT_NUMBER)
        {
          throw MaxSlotNumberError();
        }

        target_list.insert(target_list.begin() + slot_index, Attribute(value));

        target_attr.slot_number++;
      }

#define INSERT_TO_LIST(type) \
  template void Node::insert_to_list(NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX slot_index, type value);

      INSERT_TO_LIST(ATTR_CHAR)
      INSERT_TO_LIST(ATTR_UCHAR)
      INSERT_TO_LIST(ATTR_SHORT)
      INSERT_TO_LIST(ATTR_USHORT)
      INSERT_TO_LIST(ATTR_INT)
      INSERT_TO_LIST(ATTR_UINT)
      INSERT_TO_LIST(ATTR_LONG)
      INSERT_TO_LIST(ATTR_ULONG)
      INSERT_TO_LIST(ATTR_FLOAT)
      INSERT_TO_LIST(ATTR_DOUBLE)

      const char* OperationsBeforeSetupError::what() const noexcept
      {
        return "Node has not been setup.";
      }

      const char* InvalidAttributeDescError::what() const noexcept
      {
        return "Const attribute cannot be a list.";
      }

      const char* InvalidNodeIndexError::what() const noexcept
      {
        return "Node index not exist.";
      }

      const char* InvalidSlotIndexError::what() const noexcept
      {
        return "Slot index not exist.";
      }

      const char* InvalidNodeNumberError::what() const noexcept
      {
        return "Node number must be greater than 0.";
      }

      const char* InvalidAttributeTypeError::what() const noexcept
      {
        return "Attriute type note exist.";
      }

      const char* OperationsAfterSetupError::what() const noexcept
      {
        return "Cannot add attribute after setup.";
      }

      const char* OperationsOnNonListAttributeError::what() const noexcept
      {
        return "Append, clear and resize function only support for list attribute.";
      }

      const char* MaxSlotNumberError::what() const noexcept
      {
        return "Reach the max number of slot.";
      }
    }
  }
}
