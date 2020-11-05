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

      // update when removing
      void AttributeStore::update_last_index()
      {
        // 0 means no items
        while (_last_index > 1)
        {
          if (_slot_masks[_last_index - 1] == true)
          {
            break;
          }

          _last_index--;
        }
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
          auto reset_end = _last_index;

          for (auto i=0; i < _last_index; i++)
          {
            // find last that is not empty
            update_last_index();

            // false means empty slot
            if (!_slot_masks[i])
            {
              auto old_index = _last_index - 1;

              _attributes[i] = _attributes[old_index];

              _slot_masks[i] = true;
              _slot_masks[old_index] = false;

              auto key = _i2kmaping.find(old_index);

              _i2kmaping[i] = key->second;

              // update mapping to new index
              _mapping[key->second] = i;

              _i2kmaping.erase(old_index);

              // we moved last item, then move forward
              _last_index--;
            }
          }

          // in case there still some empty slot
          update_last_index();

          // set attributes we removed
          memset(&_attributes[_last_index], 0, sizeof(Attribute) * (reset_end - _last_index));

          _is_dirty = false;
        }
      }

      Attribute& AttributeStore::operator()(IDENTIFIER node_id, NODE_INDEX node_index, IDENTIFIER attr_id, SLOT_INDEX slot_index)
      {
        auto key = attr_index_key(node_id, node_index, attr_id, slot_index);
        auto index_pair = _mapping.find(key);

        if (index_pair == _mapping.end())
        {
          throw BadAttributeIndexing();
        }

        return _attributes[index_pair->second];
      }

      void AttributeStore::add_nodes(IDENTIFIER node_id, NODE_INDEX node_start_index, NODE_INDEX node_num, IDENTIFIER attr_id, SLOT_INDEX slot_num)
      {
        auto addition_num = 0;

        for (auto nindex = node_start_index; nindex < node_num; nindex++)
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

              _i2kmaping[_last_index] = key;
              _mapping[key] = _last_index;

              _last_index++;
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

      void AttributeStore::remove_node(IDENTIFIER node_id, NODE_INDEX node_index, IDENTIFIER attr_id, SLOT_INDEX slot_num)
      {
        for (auto sindex = 0; sindex < slot_num; sindex++)
        {
          auto key = attr_index_key(node_id, node_index, attr_id, sindex);

          auto attr_pair = _mapping.find(key);

          if (attr_pair != _mapping.end())
          {
            _slot_masks[attr_pair->second] = false;

            _i2kmaping.erase(attr_pair->second);
            _mapping.erase(key);

            _is_dirty = true;
          }
        }
      }

      void AttributeStore::remove_attr_slots(IDENTIFIER node_id, NODE_INDEX node_num, IDENTIFIER attr_id, SLOT_INDEX from, SLOT_INDEX stop)
      {
        for (auto nindex = 0; nindex < node_num; nindex++)
        {
          for (auto sindex = from; sindex < stop; sindex++)
          {
            auto key = attr_index_key(node_id, nindex, attr_id, sindex);

            auto attr_pair = _mapping.find(key);

            if (attr_pair != _mapping.end())
            {
              _slot_masks[attr_pair->second] = false;

              _i2kmaping.erase(attr_pair->second);
              _mapping.erase(key);

              _is_dirty = true;
            }
          }
        }
      }

      void AttributeStore::copy_to(Attribute* p, unordered_map<ULONG, size_t>& map)
      {
        // arrange before copy
        arrange();

        // do copy
        memcpy(p, &_attributes[0], _last_index * sizeof(Attribute));

        // copy the mapping
        // NOTE: this coppy will not change the index, others should consider it when using this
        for (auto iter : _mapping)
        {
          map[iter.first] = iter.second;
        }
      }

      size_t AttributeStore::capacity()
      {
        return _attributes.size();
      }

      size_t AttributeStore::last_index()
      {
        return _last_index;
      }

      size_t AttributeStore::size()
      {
        return _mapping.size();
      }

    }
  }
}
