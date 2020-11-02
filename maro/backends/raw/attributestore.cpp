#include "attributestore.h"


namespace maro
{
  namespace backends
  {
    namespace raw
    {
      inline ULONG attr_index_key(IDENTIFIER node_id, NODE_INDEX node_index, IDENTIFIER attr_id, SLOT_INDEX slot_index)
      {
        ULONG key = ULONG(slot_index) | ULONG(attr_id) << 16 | ULONG(node_index) << 32 | ULONG(node_id) << 48;

        return key;
      }

      void AttributeStore::setup(size_t size)
      {
        auto new_size = ceil(size / BITS_PER_MASK);

        _attributes.resize(new_size);
        _slot_masks.resize(new_size);
      }

      void AttributeStore::arrange()
      {
        if (_is_dirty)
        {
          for (auto i=0; i < _last_index; i++)
          {
            // false means empty slot
            if (!_slot_masks[i])
            {
              _attributes[i] = _attributes[_last_index-1];
              _last_index--;
            }
          }
        }
      }

      Attribute& AttributeStore::operator()(IDENTIFIER node_id, NODE_INDEX node_index, IDENTIFIER attr_id, SLOT_INDEX slot_index)
      {
        auto key = attr_index_key(node_id, node_index, attr_id, slot_index);
        auto index_pair = _mapping.find(key);

        //cout << "paramters" << endl;
        //cout << node_id << ", " << node_index << ", " << attr_id << ", " << slot_index << endl;
        //cout << "key" << endl;
        //cout << key << endl;

        if (index_pair == _mapping.end())
        {
          throw BadAttributeIndexing();
        }


        return _attributes[index_pair->second];
      }

      void AttributeStore::add(IDENTIFIER node_id, NODE_INDEX node_num, IDENTIFIER attr_id, SLOT_INDEX slot_num)
      {
        auto addition_num = 0;

        for (auto nindex = 0; nindex < node_num; nindex++)
        {
          for (auto sindex = 0; sindex < slot_num; sindex++)
          {
            auto key = attr_index_key(node_id, nindex, attr_id, sindex);

            // if current key exists?
            auto key_pair = _mapping.find(key);

            //
            if (key_pair == _mapping.end())
            {
              // only update for unexist slots, and these slots must be at the end.
              addition_num++;

              _mapping[key] = _last_index++;
              // we do not update slot mask here, as it make out of range
            }
          }
        }

        //extend attribute vector
        if (_last_index > _attributes.size())
        {
          _attributes.resize(_last_index * 2);
          _slot_masks.resize(_last_index * 2);
        }

        // update mask
        for (auto i = addition_num - 1; i >= 0; i--)
        {
          _slot_masks[_last_index - i - 1] = true;
        }
      }

      void AttributeStore::remove(IDENTIFIER node_id, NODE_INDEX node_index, IDENTIFIER attr_id, SLOT_INDEX slot_num)
      {
        for (auto sindex = 0; sindex < slot_num; sindex++)
        {
          auto key = attr_index_key(node_id, node_index, attr_id, sindex);

          auto attr_pair = _mapping.find(key);

          if (attr_pair != _mapping.end())
          {
            _slot_masks[attr_pair->second] = false;
            _mapping.erase(key);

            _is_dirty = true;
          }
        }
      }

      void AttributeStore::copy_to(void* p)
      {
      }

      size_t AttributeStore::size()
      {
        return _attributes.size();
      }

      size_t AttributeStore::last_index()
      {
        return _last_index;
      }

    }
  }
}
