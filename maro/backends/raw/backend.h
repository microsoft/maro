#ifndef _MARO_BACKENDS_RAW_BACKEND
#define _MARO_BACKENDS_RAW_BACKEND

#include <string>

#include "common.h"
#include "frame.h"
#include "snapshotlist.h"

using namespace std;

namespace maro
{
  namespace backends
  {
    namespace raw
    {
      class InvalidSetupState : public exception
      {};

      class Backend
      {
        Frame _frame;
        SnapshotList _snapshot;

        bool _is_setup{ false };

      public:
        IDENTIFIER add_node(string node_name, NODE_INDEX node_num);
        IDENTIFIER add_attr(IDENTIFIER node_id, string attr_name, AttrDataType attr_type, SLOT_INDEX slot_number);

        /// <summary>
        /// Setup node add attribute, after setting up, cannot add a new node type and new attribute type.
        ///
        /// But can dynamically add existing node type
        /// </summary>
        void setup();

        ATTR_BYTE get_byte(IDENTIFIER att_id, NODE_INDEX node_index, SLOT_INDEX slot_index);
        ATTR_SHORT get_short(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index);
        ATTR_INT get_int(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index);
        ATTR_LONG get_long(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index);
        ATTR_FLOAT get_float(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index);
        ATTR_DOUBLE get_double(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index);

        template<typename T>
        void set_attr_value(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index, T value);

        void enable_snapshot(USHORT number);

        void take_snapshot(INT tick);

        SnapshotResultShape prepare(IDENTIFIER node_id, INT ticks[], UINT tick_length,
          NODE_INDEX node_indices[], UINT node_length, IDENTIFIER attributes[], UINT attr_length);

        void query(QUERING_FLOAT* result, SnapshotResultShape shape);

        void dump(string path);

        void append_node(IDENTIFIER node_id, NODE_INDEX number);
        void delete_node(IDENTIFIER node_id, NODE_INDEX node_index);
        void resume_node(IDENTIFIER node_id, NODE_INDEX node_index);

        void set_attribute_slot(IDENTIFIER attr_id, SLOT_INDEX slots);

      private:
        inline void ensure_setup_state(bool expect);
      };
    }
  }
}


#endif
