#include "backend.h"

namespace maro
{
  namespace backends
  {
    namespace raw
    {
      Backend::Backend()
      {
      }

      inline void Backend::ensure_setup_state(bool expected_state)
      {
        if (_is_setup != expected_state)
        {
          throw BadSetupState();
        }
      }

      inline void Backend::ensure_node_index(NODE_INDEX cur, NodeInfo &node)
      {
        if (cur >= node.number)
        {
          throw BadNodeIndex();
        }
      }

      inline void Backend::ensure_slot_index(SLOT_INDEX cur, AttrInfo &attr)
      {
        if (cur >= attr.slots)
        {
          throw BadSlotIndex();
        }
      }

      inline void Backend::ensure_node_id(IDENTIFIER id)
      {
        if (id >= _nodes.size())
        {
          throw BadNodeIdentifier();
        }
      }

      inline void Backend::ensure_attr_id(IDENTIFIER id)
      {
        if (id >= _attrs.size())
        {
          throw BadAttributeIdentifier();
        }
      }

      inline size_t Backend::calc_attr_index(UINT frame_index, NodeInfo &node, NODE_INDEX node_index, AttrInfo &attr, SLOT_INDEX slot_index)
      {
        return (size_t)frame_index * _frame_length + (size_t)node.offset + (size_t)node.attr_number * node_index + attr.offset + slot_index;
      }

      void Backend::fill_node_indices(vector<NODE_INDEX>& node_indices, NodeInfo& node)
      {
        for (auto i = 0; i < node.number; i++)
        {
          node_indices.push_back(i);
        }
      }

      IDENTIFIER Backend::add_node(string node_name) noexcept
      {
        ensure_setup_state(false);

        auto id = _nodes.size();

        if (id >= MAX_IDENTIFIERS)
        {
          throw MaxNodeTypeNumberError();
        }

        NodeInfo node;

        node.id = id;
        node.number = 1;
        node.name = node.name;
        node.offset = 0;
        node.attr_number = 0;

        _nodes.push_back(node);

        // we can safe cast here, as we have checking before
        return IDENTIFIER(id);
      }

      IDENTIFIER Backend::add_attr(IDENTIFIER node_id, string attr_name, AttrDataType attr_type, SLOT_INDEX slot_number)
      {
        ensure_setup_state(false);

        auto id = _attrs.size();

        if (id >= MAX_IDENTIFIERS)
        {
          throw MaxAttributeTypeNumberError();
        }

        // Invalid node id will be ignored
        if (node_id >= _nodes.size())
        {
          throw BadNodeIndex();
        }

        auto &node = _nodes[node_id];

        AttrInfo attr;

        attr.id = id;
        attr.data_type = attr_type;
        attr.node_id = node_id;
        attr.name = attr_name;
        attr.slots = slot_number;
        attr.offset = node.attr_number;

        _attrs.push_back(attr);

        // we split attr slots, and treat them as different attributes
        node.attr_number += slot_number;

        return IDENTIFIER(id);
      }

      void Backend::set_node_number(IDENTIFIER node_id, NODE_INDEX number)
      {
        ensure_setup_state(false);
        ensure_node_id(node_id);

        auto &node = _nodes[node_id];

        node.number = number;
      }

      NODE_INDEX Backend::get_node_number(IDENTIFIER node_id)
      {
        ensure_node_id(node_id);

        auto &node = _nodes[node_id];

        return node.number;
      }

      USHORT Backend::get_max_snapshot_number()
      {
        return _snapshot_number;
      }

      void Backend::setup(bool enable_snapshot, USHORT snapshot_number)
      {
        ensure_setup_state(false);

        _is_snapshot_enabled = enable_snapshot;
        _snapshot_number = snapshot_number;

        // force reset snapshot_number to 0 if disabled
        if (!enable_snapshot)
        {
          _snapshot_number = 0;
        }

        size_t attr_num = 0;

        for (auto &node : _nodes)
        {
          node.offset = UINT(attr_num);

          attr_num += (size_t)node.attr_number * node.number;
        }

        // length per frame
        _frame_length = attr_num;

        attr_num *= (1 + (size_t)_snapshot_number);

        // reset to allocate enough memory
        _data.resize(attr_num);

        _is_setup = true;
      }

      void Backend::reset_frame()
      {
        // reset attribute value to 0 of current frame
        memset(&_data[0], 0, sizeof(Attribute) * _frame_length);
      }

      void Backend::reset_snapshots()
      {
        _ss_index2tick_map.clear();
        _ss_tick2index_map.clear();

        _cur_snapshot_index = 1;

        // reset attributes in snapshots to 0
        memset(&_data[_frame_length], 0, sizeof(Attribute) * (_data.size() - _frame_length));
      }

#define GET_ATTR_VALUE_BY_TYPE(m_attr_type, m_func_name)                                                     \
  m_attr_type Backend::get_##m_func_name##(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index) \
  {                                                                                                          \
    ensure_setup_state(true);                                                                                \
    ensure_attr_id(attr_id);                                                                                 \
    auto &attr = _attrs[attr_id];                                                                            \
    auto &node = _nodes[attr.node_id];                                                                       \
    ensure_node_index(node_index, node);                                                                     \
    ensure_slot_index(slot_index, attr);                                                                     \
    auto attr_index = calc_attr_index(0, node, node_index, attr, slot_index);                                \
    auto &target_attr = _data[attr_index];                                                                   \
    return target_attr.get_##m_func_name##();                                                                \
  }

      GET_ATTR_VALUE_BY_TYPE(ATTR_BYTE, byte)
      GET_ATTR_VALUE_BY_TYPE(ATTR_SHORT, short)
      GET_ATTR_VALUE_BY_TYPE(ATTR_INT, int)
      GET_ATTR_VALUE_BY_TYPE(ATTR_LONG, long)
      GET_ATTR_VALUE_BY_TYPE(ATTR_FLOAT, float)
      GET_ATTR_VALUE_BY_TYPE(ATTR_DOUBLE, double)

      template <typename T>
      void Backend::set_attr_value(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index, T value)
      {
        ensure_setup_state(true);
        ensure_attr_id(attr_id);

        auto &attr = _attrs[attr_id];
        auto &node = _nodes[attr.node_id];

        ensure_node_index(node_index, node);
        ensure_slot_index(slot_index, attr);

        auto attr_index = calc_attr_index(0, node, node_index, attr, slot_index);

        auto &target_attr = _data[attr_index];

        target_attr = value;
      }

      // make template function work in class
      template void Backend::set_attr_value<ATTR_BYTE>(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index, ATTR_BYTE value);
      template void Backend::set_attr_value<ATTR_SHORT>(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index, ATTR_SHORT value);
      template void Backend::set_attr_value<ATTR_INT>(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index, ATTR_INT value);
      template void Backend::set_attr_value<ATTR_LONG>(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index, ATTR_LONG value);
      template void Backend::set_attr_value<ATTR_FLOAT>(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index, ATTR_FLOAT value);
      template void Backend::set_attr_value<ATTR_DOUBLE>(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index, ATTR_DOUBLE value);

      void Backend::take_snapshot(INT tick)
      {
        ensure_setup_state(true);

        // we do not accept negative tick
        if (tick < 0)
        {
          throw InvalidSnapshotTick();
        }

        // make sure snapshot number will not reach the limitation
        if (_ss_tick2index_map.size() >= MAX_SNAPSHOTS)
        {
          throw MaxSnapshotError();
        }

        // try to find if tick exists
        auto internal_index_ite = _ss_tick2index_map.find(tick);

        // default write to current index
        auto target_index = _cur_snapshot_index;

        if (internal_index_ite != _ss_tick2index_map.end())
        {
          // if the tick exists, then over-write it using same internal index
          target_index = internal_index_ite->second;
        }
        else
        {
          // increase internal index
          _cur_snapshot_index += 1;

          if (_cur_snapshot_index > _snapshot_number)
          {
            // 0 is used for current frame
            _cur_snapshot_index = 1;
          }
        }

        // copy current frame data into target index
        //copy_n(_attributes.begin(), _frame_length, _attributes.begin() + ((size_t)target_index * _frame_length));
        memcpy(&_data[(size_t)target_index * _frame_length], &_data[0], _frame_length * sizeof(Attribute));

        // remove over-wrote index
        auto overwrote_index_ite = _ss_index2tick_map.find(target_index);

        if (overwrote_index_ite != _ss_index2tick_map.end())
        {
          // remove it from frame2index map to keep it clean
          _ss_tick2index_map.erase(overwrote_index_ite->second);
        }

        // update the map
        _ss_tick2index_map[tick] = target_index;
        _ss_index2tick_map[target_index] = tick;
      }

      USHORT Backend::get_valid_tick_number()
      {
        return USHORT(_ss_tick2index_map.size());
      }

      void Backend::get_ticks(INT *result)
      {
        if (result == nullptr)
        {
          return;
        }

        auto i = 0;
        for (auto iter = _ss_tick2index_map.begin(); iter != _ss_tick2index_map.end(); iter++)
        {
          result[i] = iter->first;

          i++;
        }
      }

      UINT Backend::query_one_tick_length(IDENTIFIER node_id, const NODE_INDEX node_indices[], UINT node_length, const IDENTIFIER attributes[], UINT attr_length)
      {
        ensure_node_id(node_id);

        UINT length = 0;

        auto &node = _nodes[node_id];

        vector<NODE_INDEX> _node_indices;

        // fill with node indices if no one passed
        if (node_indices == nullptr)
        {
          fill_node_indices(_node_indices, node);

          node_length = _node_indices.size();
        }

        // Choose what we need.
        const NODE_INDEX *__node_indices = node_indices == nullptr ? &_node_indices[0] : node_indices;

        // Calc the length.
        for (auto i = 0; i < node_length; i++)
        {
          auto node_index = __node_indices[i];

          ensure_node_index(node_index, node);

          for (UINT j = 0; j < attr_length; j++)
          {
            auto &attr = _attrs[attributes[j]];

            length += attr.slots;
          }
        }

        return length;
      }

      void Backend::query(ATTR_FLOAT *result, IDENTIFIER node_id, const INT ticks[], UINT ticks_length,
                          const NODE_INDEX node_indices[], UINT node_length, const IDENTIFIER attributes[], UINT attr_length)
      {
        // We do need attributes to query
        if (result == nullptr || attributes == nullptr)
        {
          return;
        }

        ensure_node_id(node_id);

        auto &node = _nodes[node_id];

        vector<INT> _ticks;

        // Prepare ticks if no one passed
        if (ticks == nullptr)
        {
          ticks_length = _ss_tick2index_map.size();

          for (auto iter = _ss_tick2index_map.begin(); iter != _ss_tick2index_map.end(); iter++)
          {
            _ticks.push_back(iter->first);
          }
        }

        vector<NODE_INDEX> _node_indices;

        if (node_indices == nullptr)
        {
          node_length = node.number;

          fill_node_indices(_node_indices, node);
        }

        // Choose what we need
        const INT *__ticks = ticks == nullptr ? &_ticks[0] : ticks;
        const NODE_INDEX *__node_indices = node_indices == nullptr ? &_node_indices[0] : node_indices;

        INT tick{0};
        // index of frame in the data array
        // 0 is current frame, others are snapshots
        UINT frame_index{1};
        UINT node_index{0};

        // Length per frame, used to padding result for invalid tick
        auto one_frame_length = query_one_tick_length(node_id, node_indices, node_length, attributes, attr_length);

        size_t ret_index = 0;

        for (auto i = 0; i < ticks_length; i++)
        {
          tick = __ticks[i];

          // skip if tick is negative, leave related parts with default value
          if (tick < 0)
          {
            ret_index += one_frame_length;

            continue;
          }

          // find if tick exists
          auto internal_index_ite = _ss_tick2index_map.find(tick);

          // skip if tick not exist, leave related parts with default value
          if (internal_index_ite == _ss_tick2index_map.end())
          {
            ret_index += one_frame_length;

            continue;
          }

          // use data at specified frame if frame exist
          frame_index = _ss_tick2index_map[tick];

          // do querying
          for (UINT j = 0; j < node_length; j++)
          {
            node_index = __node_indices[j];

            ensure_node_index(node_index, node);

            for (UINT k = 0; k < attr_length; k++)
            {
              auto &attr = _attrs[attributes[k]];

              for (SLOT_INDEX slot_index = 0; slot_index < attr.slots; slot_index++)
              {
                auto attr_index = calc_attr_index(frame_index, node, node_index, attr, slot_index);

                auto &target_attr = _data[attr_index];

                // put into result array
                *(result + ret_index) = ATTR_FLOAT(target_attr);

                ret_index += 1;
              }
            }
          }
        }
      }
    } // namespace raw
  }   // namespace backends
} // namespace maro
