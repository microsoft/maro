// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "frame.h"

namespace maro
{
  namespace backends
  {
    namespace raw
    {
      inline NODE_TYPE extract_node_type(ATTR_TYPE attr_type)
      {
        // Our ATTR_TYPE is composed with 2 parts:
        // 2 bytes: NODE_TYPE
        // 2 bytes: Attribute index in current node type
        return NODE_TYPE(attr_type >> 16);
      }

      inline void Frame::copy_from(const Frame& frame)
      {
        _nodes = frame._nodes;

        _is_setup = frame._is_setup;
      }

      inline void Frame::ensure_setup()
      {
        if (!_is_setup)
        {
          throw FrameNotSetupError();
        }
      }

      inline void Frame::ensure_node_type(NODE_TYPE node_type)
      {
        if (node_type >= _nodes.size())
        {
          throw FrameBadNodeTypeError();
        }
      }

      Frame::Frame()
      {
      }

      Frame::Frame(const Frame& frame)
      {
        copy_from(frame);
      }

      Frame& Frame::operator=(const Frame& frame)
      {
        if (this != &frame)
        {
          copy_from(frame);
        }

        return *this;
      }

      NODE_TYPE Frame::add_node(string node_name, NODE_INDEX node_number)
      {
        if (_is_setup)
        {
          throw FrameAlreadySetupError();
        }

        if (node_number == 0)
        {
          throw FrameInvalidNodeNumerError();
        }

        _nodes.emplace_back();

        // We use index as node type for easily querying.
        NODE_TYPE node_type = _nodes.size() - 1;

        auto& node = _nodes[node_type];

        node.set_name(node_name);
        node.set_type(node_type);
        node.set_defined_number(node_number);

        return node_type;
      }

      ATTR_TYPE Frame::add_attr(NODE_TYPE node_type, string attr_name, AttrDataType data_type,
        SLOT_INDEX slot_number, bool is_const, bool is_list)
      {
        if (_is_setup)
        {
          throw FrameAlreadySetupError();
        }

        ensure_node_type(node_type);

        auto& node = _nodes[node_type];

        return node.add_attr(attr_name, data_type, slot_number, is_const, is_list);
      }

      Node& Frame::get_node(NODE_TYPE node_type)
      {
        ensure_setup();
        ensure_node_type(node_type);

        return _nodes[node_type];
      }

      void Frame::append_node(NODE_TYPE node_type, NODE_INDEX node_number)
      {
        auto& node = get_node(node_type);

        node.append_nodes(node_number);
      }

      void Frame::remove_node(NODE_TYPE node_type, NODE_INDEX node_index)
      {
        auto& node = get_node(node_type);
        node.remove_node(node_index);
      }

      void Frame::resume_node(NODE_TYPE node_type, NODE_INDEX node_index)
      {
        auto& node = get_node(node_type);
        node.resume_node(node_index);
      }

      void Frame::clear_list(NODE_INDEX node_index, ATTR_TYPE attr_type)
      {
        NODE_TYPE node_type = extract_node_type(attr_type);

        auto& node = get_node(node_type);

        node.clear_list(node_index, attr_type);
      }

      void Frame::resize_list(NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX new_size)
      {
        NODE_TYPE node_type = extract_node_type(attr_type);

        auto& node = get_node(node_type);

        node.resize_list(node_index, attr_type, new_size);
      }

      void Frame::setup()
      {
        if (_is_setup)
        {
          return;
        }

        for (auto& node : _nodes)
        {
          node.setup();
        }

        _is_setup = true;
      }

      void Frame::reset()
      {
        ensure_setup();

        for (auto& node : _nodes)
        {
          node.reset();
        }
      }

      bool Frame::is_node_exist(NODE_TYPE node_type) const noexcept
      {
        return node_type < _nodes.size();
      }

      template<typename T>
      typename Attribute_Trait<T>::type Frame::get_value(NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX slot_index)
      {
        NODE_TYPE node_type = extract_node_type(attr_type);

        auto& node = get_node(node_type);

        auto& target_attr = node.get_attr(node_index, attr_type, slot_index);

        return target_attr.get_value<T>();
      }

#define ATTRIBUTE_GETTER(type) \
  template type Frame::get_value<type>(NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX slot_index);

      ATTRIBUTE_GETTER(ATTR_CHAR)
      ATTRIBUTE_GETTER(ATTR_UCHAR)
      ATTRIBUTE_GETTER(ATTR_SHORT)
      ATTRIBUTE_GETTER(ATTR_USHORT)
      ATTRIBUTE_GETTER(ATTR_INT)
      ATTRIBUTE_GETTER(ATTR_UINT)
      ATTRIBUTE_GETTER(ATTR_LONG)
      ATTRIBUTE_GETTER(ATTR_ULONG)
      ATTRIBUTE_GETTER(ATTR_FLOAT)
      ATTRIBUTE_GETTER(ATTR_DOUBLE)

      template<typename T>
      void Frame::set_value(NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX slot_index, T value)
      {
        NODE_TYPE node_type = extract_node_type(attr_type);

        auto& node = get_node(node_type);

        auto& target_attr = node.get_attr(node_index, attr_type, slot_index);

        target_attr = T(value);
      }

#define ATTRIBUTE_SETTER(type) \
  template void Frame::set_value(NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX slot_index, type value);

      ATTRIBUTE_SETTER(ATTR_CHAR)
      ATTRIBUTE_SETTER(ATTR_UCHAR)
      ATTRIBUTE_SETTER(ATTR_SHORT)
      ATTRIBUTE_SETTER(ATTR_USHORT)
      ATTRIBUTE_SETTER(ATTR_INT)
      ATTRIBUTE_SETTER(ATTR_UINT)
      ATTRIBUTE_SETTER(ATTR_LONG)
      ATTRIBUTE_SETTER(ATTR_ULONG)
      ATTRIBUTE_SETTER(ATTR_FLOAT)
      ATTRIBUTE_SETTER(ATTR_DOUBLE)

      template<typename T>
      void Frame::append_to_list(NODE_INDEX node_index, ATTR_TYPE attr_type, T value)
      {
        NODE_TYPE node_type = extract_node_type(attr_type);

        auto& node = get_node(node_type);

        node.append_to_list<T>(node_index, attr_type, value);
      }

#define ATTRIBUTE_APPENDER(type) \
  template void Frame::append_to_list(NODE_INDEX node_index, ATTR_TYPE attr_type, type value);

      ATTRIBUTE_APPENDER(ATTR_CHAR)
      ATTRIBUTE_APPENDER(ATTR_UCHAR)
      ATTRIBUTE_APPENDER(ATTR_SHORT)
      ATTRIBUTE_APPENDER(ATTR_USHORT)
      ATTRIBUTE_APPENDER(ATTR_INT)
      ATTRIBUTE_APPENDER(ATTR_UINT)
      ATTRIBUTE_APPENDER(ATTR_LONG)
      ATTRIBUTE_APPENDER(ATTR_ULONG)
      ATTRIBUTE_APPENDER(ATTR_FLOAT)
      ATTRIBUTE_APPENDER(ATTR_DOUBLE)

      void Frame::remove_from_list(NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX slot_index)
      {
        NODE_TYPE node_type = extract_node_type(attr_type);

        auto& node = get_node(node_type);

        node.remove_from_list(node_index, attr_type, slot_index);
      }

      template<typename T>
      void Frame::insert_to_list(NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX slot_index, T value)
      {
        NODE_TYPE node_type = extract_node_type(attr_type);

        auto& node = get_node(node_type);

        node.insert_to_list(node_index, attr_type, slot_index, value);
      }

#define ATTRIBUTE_INSERTER(type) \
  template void Frame::insert_to_list(NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX slot_index, type value);

      ATTRIBUTE_INSERTER(ATTR_CHAR)
      ATTRIBUTE_INSERTER(ATTR_UCHAR)
      ATTRIBUTE_INSERTER(ATTR_SHORT)
      ATTRIBUTE_INSERTER(ATTR_USHORT)
      ATTRIBUTE_INSERTER(ATTR_INT)
      ATTRIBUTE_INSERTER(ATTR_UINT)
      ATTRIBUTE_INSERTER(ATTR_LONG)
      ATTRIBUTE_INSERTER(ATTR_ULONG)
      ATTRIBUTE_INSERTER(ATTR_FLOAT)
      ATTRIBUTE_INSERTER(ATTR_DOUBLE)

      void Frame::write_attribute(ofstream &file, NODE_INDEX node_index, ATTR_TYPE attr_id, SLOT_INDEX slot_index)
      {

      }

      void Frame::dump(string folder)
      {
        // for dump, we will save for each node, named as "node_<node_name>.csv"
        // content of the csv will follow padans' output that list will be wrapped into a "[]",
        for (auto& node : _nodes)
        {
          auto output_path = folder + "/" + "node_" + node._name + ".csv";

          ofstream file(output_path);

          // Write header - first column.
          file << "node_index";

          // Futhure columns (attribute name).
          for(auto& attr_def : node._attribute_definitions)
          {
            file << "," << attr_def.name;
          }

          // End of header.
          file << "\n";

          // Write for each node instance.
          for (NODE_INDEX node_index = 0; node_index < node._max_node_number; node_index++)
          {
            // Ignore deleted node instance.
            if(!node.is_node_alive(node_index))
            {
              continue;
            }

            // Row - node index.
            file << node_index;

            for (auto& attr_def : node._attribute_definitions)
            {
              if (!attr_def.is_list && attr_def.slot_number == 1)
              {
                file << ",";

                auto& attr = node.get_attr(node_index, attr_def.attr_type, 0);

                file << QUERY_FLOAT(attr);
              }
              else
              {
                // List start.
                file << ",\"[";

                auto slot_number = node.get_slot_number(node_index, attr_def.attr_type);

                for (SLOT_INDEX slot_index = 0; slot_index < slot_number; slot_index++)
                {
                  auto& attr = node.get_attr(node_index, attr_def.attr_type, 0);

                  file << QUERY_FLOAT(attr);

                  file << ",";
                }

                // List end.
                file << "]\"";
              }
            }

            // end of row
            file << "\n";
          }

          file.close();
        }
      }

      const char* FrameNotSetupError::what() const noexcept
      {
        return "Frame has not been setup.";
      }

      const char* FrameAlreadySetupError::what() const noexcept
      {
        return "Cannot add new node or attribute type after setting up.";
      }

      const char* FrameBadNodeTypeError::what() const noexcept
      {
        return "Not exist node type.";
      }

      const char* FrameBadAttributeTypeError::what() const noexcept
      {
        return "Not exist attribute type.";
      }

      const char* FrameInvalidNodeNumerError::what() const noexcept
      {
        return "Node number must be greater than 0.";
      }
    }
  }
}
