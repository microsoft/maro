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

      IDENTIFIER Backend::add_node(string node_name, USHORT node_num)
      {
        ensure_setup_state(false);

        return _frame.new_node(node_name, node_num);
      }

      IDENTIFIER Backend::add_attr(IDENTIFIER node_id, string attr_name, AttrDataType attr_type, SLOT_INDEX slot_number)
      {
        ensure_setup_state(false);

        return _frame.new_attr(node_id, attr_name, attr_type, slot_number);
      }

      void Backend::setup()
      {
        ensure_setup_state(false);

        _frame.setup();

        _snapshot.set_frame(&_frame);

        _is_setup = true;
      }

      ATTR_BYTE Backend::get_byte(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index)
      {
        ensure_setup_state(true);

        return _frame(node_index, attr_id, slot_index).get_byte();
      }

      ATTR_SHORT Backend::get_short(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index)
      {
        ensure_setup_state(true);

        return _frame(node_index, attr_id, slot_index).get_short();
      }

      ATTR_INT Backend::get_int(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index)
      {
        ensure_setup_state(true);

        return _frame(node_index, attr_id, slot_index).get_int();
      }

      ATTR_LONG Backend::get_long(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index)
      {
        ensure_setup_state(true);

        return _frame(node_index, attr_id, slot_index).get_long();
      }

      ATTR_FLOAT Backend::get_float(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index)
      {
        ensure_setup_state(true);

        return _frame(node_index, attr_id, slot_index).get_float();
      }

      ATTR_DOUBLE Backend::get_double(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index)
      {
        ensure_setup_state(true);

        return _frame(node_index, attr_id, slot_index).get_double();
      }

      void Backend::enable_snapshot(USHORT number)
      {
        ensure_setup_state(false);

        _snapshot.set_max_size(number);
      }

      void Backend::take_snapshot(INT tick)
      {
        ensure_setup_state(true);

        _snapshot.take_snapshot(tick);
      }

      SnapshotResultShape Backend::prepare(IDENTIFIER node_id, INT ticks[], UINT tick_length,
        NODE_INDEX node_indices[], UINT node_length, IDENTIFIER attributes[], UINT attr_length)
      {
        ensure_setup_state(true);

        return _snapshot.prepare(node_id, ticks, tick_length, node_indices, node_length, attributes, attr_length);
      }

      void Backend::query(QUERING_FLOAT* result, SnapshotResultShape shape)
      {
        ensure_setup_state(true);

        return _snapshot.query(result, shape);
      }

      void Backend::dump(string path)
      {
      }

      template<typename T>
      void Backend::set_attr_value(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index, T value)
      {
        ensure_setup_state(true);

        auto& attr = _frame(node_index, attr_id, slot_index);

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
    }
  }
}
