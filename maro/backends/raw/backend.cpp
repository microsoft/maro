#include "backend.h"

namespace maro
{
  namespace backends
  {
    namespace raw
    {
      inline void Backend::ensure_setup_state(bool expect)
      {
        if (_is_setup != expect)
        {
          throw InvalidSetupState();
        }
      }

      IDENTIFIER Backend::add_node(string node_name, NODE_INDEX node_num)
      {
        ensure_setup_state(false);

        return _frame.new_node(node_name, node_num);
      }

      IDENTIFIER Backend::add_attr(IDENTIFIER node_id, string attr_name, AttrDataType attr_type, SLOT_INDEX slot_number)
      {
        ensure_setup_state(false);

        return _frame.new_attr(node_id, attr_name, attr_type, slot_number);
      }

      void Backend::setup(bool enable_snapshot, USHORT snapshot_number)
      {
        ensure_setup_state(false);

        _frame.setup();

        if (enable_snapshot)
        {
          _snapshot.set_frame(&_frame);

          _snapshot.set_max_size(snapshot_number);
        }

        _is_snapshot_enabled = enable_snapshot;
        _is_setup = true;
      }

#define ATTR_GETTER(m_return_type, m_func_type)                                                              \
  m_return_type Backend::get_##m_func_type(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index) \
  {                                                                                                          \
    ensure_setup_state(true);                                                                                \
    return _frame(node_index, attr_id, slot_index).get_##m_func_type();                                      \
  }

      ATTR_GETTER(ATTR_BYTE, byte);
      ATTR_GETTER(ATTR_SHORT, short);
      ATTR_GETTER(ATTR_INT, int);
      ATTR_GETTER(ATTR_LONG, long);
      ATTR_GETTER(ATTR_FLOAT, float);
      ATTR_GETTER(ATTR_DOUBLE, double);

      void Backend::take_snapshot(INT tick)
      {
        ensure_setup_state(true);

        _snapshot.take_snapshot(tick);
      }

      void Backend::reset_frame()
      {
        _frame.reset();
      }

      void Backend::reset_snapshots()
      {
        _snapshot.reset();
      }

      SnapshotResultShape Backend::prepare(IDENTIFIER node_id, INT ticks[], UINT tick_length,
                            NODE_INDEX node_indices[], UINT node_length, IDENTIFIER attributes[], UINT attr_length)
      {
        ensure_setup_state(true);

        return _snapshot.prepare(node_id, ticks, tick_length, node_indices, node_length, attributes, attr_length);
      }

      void Backend::query(QUERING_FLOAT *result, SnapshotResultShape shape)
      {
        ensure_setup_state(true);

        return _snapshot.query(result, shape);
      }

      USHORT Backend::get_max_snapshot_number()
      {
        return _snapshot.max_size();
      }

      USHORT Backend::get_valid_tick_number()
      {
        return _snapshot.size();
      }

      void Backend::get_ticks(INT *result)
      {
        _snapshot.get_ticks(result);
      }

      void Backend::dump_current_frame(string path)
      {
        _frame.dump(path);
      }

      void Backend::dump_snapshots(string path)
      {
        _snapshot.dump(path);
      }

      template <typename T>
      void Backend::set_attr_value(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index, T value)
      {
        ensure_setup_state(true);

        auto &attr = _frame(node_index, attr_id, slot_index);

        attr = value;
      }

      template void Backend::set_attr_value(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index, ATTR_BYTE value);
      template void Backend::set_attr_value(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index, ATTR_SHORT value);
      template void Backend::set_attr_value(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index, ATTR_INT value);
      template void Backend::set_attr_value(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index, ATTR_LONG value);
      template void Backend::set_attr_value(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index, ATTR_FLOAT value);
      template void Backend::set_attr_value(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index, ATTR_DOUBLE value);

      void Backend::append_node(IDENTIFIER node_id, NODE_INDEX number)
      {
        ensure_setup_state(true);

        _frame.append_nodes(node_id, number);
      }

      void Backend::delete_node(IDENTIFIER node_id, NODE_INDEX node_index)
      {
        ensure_setup_state(true);

        _frame.remove_node(node_id, node_index);
      }

      void Backend::resume_node(IDENTIFIER node_id, NODE_INDEX node_index)
      {
        ensure_setup_state(true);

        _frame.resume_node(node_id, node_index);
      }

      void Backend::set_attribute_slot(IDENTIFIER attr_id, SLOT_INDEX slots)
      {
        ensure_setup_state(true);

        _frame.set_attr_slot(attr_id, slots);
      }

      USHORT Backend::get_node_number(IDENTIFIER node_id)
      {
        return _frame.get_node_number(node_id);
      }

      USHORT Backend::get_slots_number(IDENTIFIER attr_id)
      {
        return _frame.get_slots_number(attr_id);
      }
    } // namespace raw
  }   // namespace backends
} // namespace maro
