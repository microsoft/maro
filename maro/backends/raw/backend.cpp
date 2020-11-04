#include "backend.h"

namespace maro
{
  namespace backends
  {
    namespace raw
    {
      IDENTIFIER Backend::reg_node(string name, NODE_INDEX number)
      {
        return IDENTIFIER();
      }
      IDENTIFIER Backend::reg_attr(IDENTIFIER node_id, string name, AttrDataType type, SLOT_INDEX slots)
      {
        return IDENTIFIER();
      }
      void Backend::setup()
      {
      }
    }
  }
}
