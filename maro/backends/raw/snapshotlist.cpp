// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "snapshotlist.h"

namespace maro
{
  namespace backends
  {
    namespace raw
    {
      inline void SnapshotList::ensure_cur_frame()
      {
        if (_cur_frame == nullptr)
        {
          throw SnapshotInvalidFrameStateError();
        }
      }

      inline void SnapshotList::ensure_max_size()
      {
        if (_max_size == 0)
        {
          throw SnapshotSizeError();
        }
      }

      void SnapshotList::set_max_size(USHORT max_size)
      {
        _max_size = max_size;

        ensure_max_size();
      }

      void SnapshotList::setup(Frame* frame)
      {
        _cur_frame = frame;

        ensure_cur_frame();
      }

      void SnapshotList::take_snapshot(int tick)
      {
        ensure_max_size();
        ensure_cur_frame();

        // Try to remove exist tick.
        _snapshots.erase(tick);

        // Remove oldest one if we reach the max size limitation.
        if (_snapshots.size() > 0 && _snapshots.size() >= _max_size)
        {
          _snapshots.erase(_snapshots.begin());
        }

        // Copy current frame.
        _snapshots[tick] = *_cur_frame;
      }

      UINT SnapshotList::size() const noexcept
      {
        return _snapshots.size();
      }

      UINT SnapshotList::max_size() const noexcept
      {
        return _max_size;
      }

      NODE_INDEX SnapshotList::get_max_node_number(NODE_TYPE node_type) const
      {
        auto& cur_node = _cur_frame->get_node(node_type);

        return cur_node.get_max_number();
      }

      void SnapshotList::reset()
      {
        _snapshots.clear();
      }

      void SnapshotList::get_ticks(int* result) const
      {
        if (result == nullptr)
        {
          throw SnapshotQueryResultPtrNullError();
        }

        auto i = 0;
        for (auto& iter : _snapshots)
        {
          result[i] = iter.first;

          i++;
        }
      }

      SnapshotQueryResultShape SnapshotList::prepare(NODE_TYPE node_type, int ticks[], UINT tick_length, NODE_INDEX node_indices[], UINT node_length, ATTR_TYPE attributes[], UINT attr_length)
      {
        SnapshotQueryResultShape shape;

        if (attributes == nullptr)
        {
          throw SnapshotQueryNoAttributesError();
        }

        // Node in current frame, used to get attribute definition.
        auto& cur_node = _cur_frame->get_node(node_type);
        auto first_attr_type = attributes[0];
        auto& attr_definition = cur_node.get_attr_definition(first_attr_type);

        // We use first attribute determine the type of current querying.
        _query_parameters.is_list = attr_definition.is_list;

        shape.max_node_number = node_indices == nullptr ? cur_node.get_max_number() : node_length;
        shape.tick_number = ticks == nullptr ? _snapshots.size() : tick_length;

        if (!_query_parameters.is_list)
        {
          // If it is not a list attriubte, then accept all attribute except list .
          for (UINT attr_index = 0; attr_index < attr_length; attr_index++)
          {
            auto attr_type = attributes[attr_index];
            auto& attr_def = cur_node.get_attr_definition(attr_type);

            if (attr_def.is_list)
            {
              // warning and ignore it
              cerr << "Ignore list attribute: " << attr_def.name << " for fixed size attribute querying." << endl;
              continue;
            }

            shape.attr_number++;
            shape.max_slot_number = max(attr_def.slot_number, shape.max_slot_number);
          }
        }
        else
        {
          // If it is a list attribute, then just use first one as querying attribute,
          // we only support query 1 list attribute (1st one) for 1 node at 1 tick each time to reduce too much padding.

          // Make sure we have at least one tick.
          if (_snapshots.size() == 0)
          {
            throw SnapshotQueryNoSnapshotsError();
          }

          // There must be 1 node index for list attribute querying.
          if (node_indices == nullptr)
          {
            throw SnapshotListQueryNoNodeIndexError();
          }

          // 1 tick, 1 node and 1 attribute for list attribute querying.
          shape.attr_number = 1;
          shape.tick_number = 1;
          shape.max_node_number = 1;

          // Use first tick in parameter, or latest tick in snapshot.
          int tick = ticks == nullptr ? _snapshots.rbegin()->first : ticks[0];
          auto target_node_index = node_indices[0];

          // Check if tick exist.
          auto target_tick_pair = _snapshots.find(tick);

          if (target_tick_pair == _snapshots.end())
          {
            throw SnapshotQueryNoSnapshotsError();
          }

          auto& snapshot = target_tick_pair->second;
          auto& history_node = snapshot.get_node(node_type);

          // Check if the node index exist.
          if (!history_node.is_node_alive(target_node_index))
          {
            throw SnapshotListQueryNoNodeIndexError();
          }

          shape.max_slot_number = history_node.get_slot_number(target_node_index, first_attr_type);
        }

        _query_parameters.ticks = ticks;
        _query_parameters.node_indices = node_indices;
        _query_parameters.attributes = attributes;

        _query_parameters.node_type = node_type;

        _query_parameters.max_slot_number = shape.max_slot_number;
        _query_parameters.attr_length = shape.attr_number;
        _query_parameters.tick_length = shape.tick_number;
        _query_parameters.node_length = shape.max_node_number;

        _is_prepared = true;

        return shape;
      }

      void SnapshotList::query(QUERY_FLOAT* result)
      {
        if (!_is_prepared)
        {
          throw SnapshotQueryNotPreparedError();
        }

        _is_prepared = false;

        if (!_query_parameters.is_list)
        {
          query_for_normal(result);
        }
        else
        {
          query_for_list(result);
        }

        _query_parameters.reset();
      }

      void SnapshotList::query_for_list(QUERY_FLOAT* result)
      {
        auto* ticks = _query_parameters.ticks;
        auto max_slot_number = _query_parameters.max_slot_number;
        auto tick = ticks == nullptr ? _snapshots.rbegin()->first : ticks[0];
        auto node_index = _query_parameters.node_indices[0];
        auto attr_type = _query_parameters.attributes[0];

        // Go through all slots.
        for (UINT i = 0; i < max_slot_number; i++)
        {
          auto& attr = get_attr(tick, node_index, attr_type, i);

          // Ignore nan for now, use default value from outside.
          if (!attr.is_nan())
          {
            result[i] = QUERY_FLOAT(attr);
          }
        }
      }

      void SnapshotList::query_for_normal(QUERY_FLOAT* result)
      {
        auto node_type = _query_parameters.node_type;

        // Node in current frame, used to get attribute defition and const value.
        auto& node = _cur_frame->get_node(node_type);

        auto* ticks = _query_parameters.ticks;
        auto* node_indices = _query_parameters.node_indices;
        auto* attrs = _query_parameters.attributes;
        auto tick_length = _query_parameters.tick_length;
        auto node_length = _query_parameters.node_length;
        auto attr_length = _query_parameters.attr_length;
        auto max_slot_number = _query_parameters.max_slot_number;

        vector<int> _ticks;

        // Prepare ticks if no one provided.
        if (_query_parameters.ticks == nullptr)
        {
          tick_length = _snapshots.size();

          for (auto& iter : _snapshots)
          {
            _ticks.push_back(iter.first);
          }
        }

        vector<NODE_INDEX> _node_indices;

        // Prepare node indices if no one provided.
        if (node_indices == nullptr)
        {
          node_length = node.get_max_number();

          for (UINT i = 0; i < node_length; i++)
          {
            _node_indices.push_back(i);
          }
        }

        const int* __ticks = ticks == nullptr ? &_ticks[0] : ticks;
        const NODE_INDEX* __node_indices = node_indices == nullptr ? &_node_indices[0] : node_indices;

        // Index in result list.
        auto result_index = 0;

        // Go through by tick -> node -> attribute -> slot.
        for (UINT i = 0; i < tick_length; i++)
        {
          auto tick = __ticks[i];

          for (UINT j = 0; j < node_length; j++)
          {
            auto node_index = __node_indices[j];

            for (UINT k = 0; k < attr_length; k++)
            {
              auto attr_type = attrs[k];

              for (SLOT_INDEX slot_index = 0; slot_index < max_slot_number; slot_index++)
              {
                auto& attr = get_attr(tick, node_index, attr_type, slot_index);

                if (!attr.is_nan())
                {
                  result[result_index] = ATTR_FLOAT(attr);
                }

                result_index++;
              }
            }
          }
        }
      }

      void SnapshotList::cancel_query() noexcept
      {
        _is_prepared = false;
        _query_parameters.reset();
      }

      Attribute& SnapshotList::get_attr(int tick, NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX slot_index) noexcept
      {
        NODE_TYPE node_type = extract_node_type(attr_type);

        // check if node exist
        if(!_cur_frame->is_node_exist(node_type))
        {
          return _nan_attr;
        }

        auto& cur_node = _cur_frame->get_node(node_type);
        const auto& attr_def = cur_node.get_attr_definition(attr_type);

        // Check slot index for non-list attribute.
        if (!attr_def.is_list && slot_index >= attr_def.slot_number)
        {
            return _nan_attr;
        }

        // If it is a const attribute, retrieve from const block,
        // we do not care if tick exist for const attribute.
        if (attr_def.is_const)
        {
          return cur_node.get_attr(node_index, attr_type, slot_index);
        }

        auto target_tick_pair = _snapshots.find(tick);

        // Check if tick valid.
        if (target_tick_pair == _snapshots.end())
        {
          return _nan_attr;
        }

        auto& snapshot = target_tick_pair->second;
        auto& history_node = snapshot.get_node(node_type);

        // Check if node index valid.
        if (node_index >= history_node._max_node_number || !history_node.is_node_alive(node_index))
        {
          return _nan_attr;
        }

        if (attr_def.is_list)
        {
          auto attr_offset = compose_attr_offset_in_node(node_index, history_node._dynamic_size_per_node, attr_def.offset);
          auto& target_attr = history_node._dynamic_block[attr_offset];

          const auto list_index = target_attr.get_value<ATTR_UINT>();

          auto& target_list = history_node._list_store[list_index];

          // Check slot for list attribute.
          if (slot_index >= target_list.size())
          {
            return _nan_attr;
          }

          return target_list[slot_index];
        }

        auto attr_offset = compose_attr_offset_in_node(node_index, history_node._dynamic_size_per_node, attr_def.offset, slot_index);

        return history_node._dynamic_block[attr_offset];
      }

      void SnapshotList::SnapshotQueryParameters::reset()
      {
        ticks = nullptr;
        attributes = nullptr;
        node_indices = nullptr;

        tick_length = 0;
        node_length = 0;
        attr_length = 0;
        max_slot_number = 0;

        is_list = false;
      }

      inline void SnapshotList::write_attribute(ofstream &file, int tick, NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX slot_index)
      {
        auto &attr = get_attr(tick, node_index, attr_type, slot_index);

        if (attr.is_nan())
        {
          file << "nan";
        }
        else
        {
          file << ATTR_FLOAT(attr);
        }
      }

      void SnapshotList::dump(string path)
      {
        for (auto& node : _cur_frame->_nodes)
        {
          auto full_path = path + "/" + "snapshots_" + node._name + ".csv";

          ofstream file(full_path);

          // Headers.
          file << "tick,node_index";

          for(auto& attr_def : node._attribute_definitions)
          {
            file << "," << attr_def.name;
          }

          // End of Headers.
          file << "\n";

          // Rows.
          for(auto& snapshot_iter : _snapshots)
          {
            auto tick = snapshot_iter.first;
            auto& snapshot = snapshot_iter.second;
            auto& history_node = snapshot.get_node(node._type);

            for (NODE_INDEX node_index = 0; node_index < node._max_node_number; node_index++)
            {
              // ignore deleted node
              if(!history_node.is_node_alive(node_index))
              {
                continue;
              }

              file << tick << "," << node_index;

              for(auto& attr_def : node._attribute_definitions)
              {
                if(!attr_def.is_list && attr_def.slot_number == 1)
                {
                  file << ",";

                  write_attribute(file, tick, node_index, attr_def.attr_type, 0);
                }
                else
                {
                  file << ",\"[";

                  auto slot_number = history_node.get_slot_number(node_index, attr_def.attr_type);

                  for(SLOT_INDEX slot_index = 0; slot_index < slot_number; slot_index++)
                  {
                    write_attribute(file, tick, node._type, attr_def.attr_type, slot_index);

                    file << ",";
                  }

                  file << "]\"";
                }
              }

              file << "\n";
            }
          }

          file.close();
        }
      }

      const char* SnapshotTickError::what() const noexcept
      {
        return "Invalid tick to take snapshot, same tick must be used sequentially.";
      }

      const char* SnapshotSizeError::what() const noexcept
      {
        return "Invalid snapshot list max size, it must be larger than 0.";
      }

      const char* SnapshotQueryNotPreparedError::what() const noexcept
      {
        return "Query must be after prepare function.";
      }

      const char* SnapshotQueryNoAttributesError::what() const noexcept
      {
        return "Attribute list for query should contain at least 1.";
      }

      const char* SnapshotInvalidFrameStateError::what() const noexcept
      {
        return "Not set frame before operations.";
      }

      const char* SnapshotQueryResultPtrNullError::what() const noexcept
      {
        return "Result pointer is NULL.";
      }

      const char* SnapshotQueryInvalidTickError::what() const noexcept
      {
        return "Only support one tick to query for list attribute, and the tick must exist.";
      }

      const char* SnapshotQueryNoSnapshotsError::what() const noexcept
      {
        return "List attribute querying need at lease one snapshot, it does not support invalid tick padding.";
      }

      const char* SnapshotListQueryNoNodeIndexError::what() const noexcept
      {
        return "List attribute querying need one alive node index.";
      }
    }
  }
}
