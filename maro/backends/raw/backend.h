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
      class Backend
      {
        Frame _frame;
        SnapshotList _snapshot;

      public:
        IDENTIFIER reg_node(string name, NODE_INDEX number);

        IDENTIFIER reg_attr(IDENTIFIER node_id, string name, AttrDataType type, SLOT_INDEX slots=1);

        void setup();


        ATTR_BYTE get_byte(IDENTIFIER att_id, NODE_INDEX node_index, SLOT_INDEX slot_index);
        ATTR_SHORT get_short(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index);
        ATTR_INT get_int(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index);
        ATTR_LONG get_long(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index);
        ATTR_FLOAT get_float(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index);
        ATTR_DOUBLE get_double(IDENTIFIER attr_id, NODE_INDEX node_index, SLOT_INDEX slot_index);

        void set_value();


        void delete_node(IDENTIFIER node_id, NODE_INDEX node_index);

        void append_node(IDENTIFIER node_id, NODE_INDEX number);

        void set_attribute_slot(IDENTIFIER attr_id, SLOT_INDEX slots);



        void enable_snapshot(USHORT number);

        void take_snapshot(INT tick);

        void dump(string path);

      };
    }
  }
}


#endif
