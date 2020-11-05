#include "snapshotlist.h"

namespace maro
{
  namespace backends
  {
    namespace raw
    {
      void SnapshotList::set_max_size(USHORT max_size)
      {
        if (max_size == 0)
        {
          throw InvalidSnapshotSize();
        }

        if (_max_size == 0)
        {
          _max_size = max_size;
        }
      }

      void SnapshotList::take_snapshot(INT tick, AttributeStore& frame_attr_store)
      {
        ensure_max_size();

        // To make it easy to implement, we do not support over-write exist tick at any time,
        // tick can onlly be over-wrote if last one is same tick

        // arrange before take snapshot
        frame_attr_store.arrange();

        auto snapshot_size = frame_attr_store.size();

        {
          // 1. check tick exist
          auto tick_pair = _tick2index_map.find(tick);

          // tick exist
          if (tick_pair != _tick2index_map.end())
          {
            // then check if last tick is same
            if (_last_tick != tick)
            {
              throw InvalidSnapshotTick();
            }

            // for exist tick, it has 2 situation
            // 1. at the end:
            //    we can just set _end_index to start of this tick
            // 2. just before empty slots
            //    we just move first empty slot index to start of this tick, and its length to empty slots length

            auto exist_tick_index = tick_pair->second;
            auto exist_tick_length = _tick2size_map.find(tick)->second;

            // remove info about this tick
            _tick2index_map.erase(tick);
            _tick2size_map.erase(tick);
            _tick_attr_map.erase(tick);

            // is this tick at the end?
            if (exist_tick_index + exist_tick_length == _end_index)
            {
              _end_index = exist_tick_index;
            }
            else
            {
              _first_empty_slot_index = exist_tick_index;
              _empty_slots_length += exist_tick_length;
            }

            _cur_snapshot_num--;
          }
        }

        _cur_snapshot_num++;

        if (_cur_snapshot_num > _max_size)
        {
          // Do overlap

          // find oldest tick to delete
          auto oldest_item = _tick2index_map.begin();
          auto oldest_tick = oldest_item->first;
          auto oldest_index = oldest_item->second;
          auto oldest_size = _tick2size_map.find(oldest_tick)->second;

          /// remove from mappings
          _tick2index_map.erase(oldest_tick);
          _tick2size_map.erase(oldest_tick);
          _tick_attr_map.erase(oldest_tick);

          // update empty slots area flags

          // if not empty slots in the middle, then use current as first
          if (_empty_slots_length == 0)
          {
            _first_empty_slot_index = oldest_index;
            _empty_slots_length = oldest_size;
          }
          else
          {
            // or it must be right after current empty slots
            _empty_slots_length += oldest_size;
          }

          // if remaining empty slots enough?
          if (_empty_slots_length >= snapshot_size)
          {
            write_to_empty_slots(frame_attr_store, tick);
          }
          else
          {
            // append to the end
            append_to_end(frame_attr_store, tick);
          }
        }
        else
        {
          // append
          append_to_end(frame_attr_store, tick);
        }

      }

      void SnapshotList::append_to_end(AttributeStore& frame_attr_store, INT tick)
      {
        auto snapshot_size = frame_attr_store.size();

        // prepare attribute store to make sure we can hold all
        if (_end_index + snapshot_size < _attr_store.size())
        {
          _attr_store.resize(_end_index + snapshot_size);
        }

        // copy
        auto mapping = unordered_map<ULONG, size_t>();

        frame_attr_store.copy_to(&_attr_store[_end_index], mapping);

        _tick_attr_map[tick] = mapping;
        _tick2size_map[tick] = snapshot_size;
        _tick2index_map[tick] = _end_index;

        _end_index += snapshot_size;
      }

      void SnapshotList::write_to_empty_slots(AttributeStore& frame_attr_store, INT tick)
      {
        auto snapshot_size = frame_attr_store.size();

        // write to here
        auto mapping = unordered_map<ULONG, size_t>();

        frame_attr_store.copy_to(&_attr_store[_first_empty_slot_index], mapping);

        _tick_attr_map[tick] = mapping;
        _tick2index_map[tick] = _first_empty_slot_index;
        _tick2size_map[tick] = snapshot_size;

        _first_empty_slot_index += snapshot_size;
        _empty_slots_length -= snapshot_size;
      }

      inline void SnapshotList::ensure_max_size()
      {
        if (_max_size == 0)
        {
          throw InvalidSnapshotSize();
        }
      }

      void SnapshotList::query(QUERING_FLOAT* result, IDENTIFIER node_id, INT ticks[], UINT tick_length, NODE_INDEX node_indices[], UINT node_length, IDENTIFIER attributes, UINT attr_length)
      {
        ensure_max_size();
      }

      USHORT SnapshotList::size()
      {
        return _max_size;
      }
    }
  }
}
