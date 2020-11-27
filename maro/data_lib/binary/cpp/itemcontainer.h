// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef _MARO_DATALIB_ITEM_CONTAINER_
#define _MARO_DATALIB_ITEM_CONTAINER_

#include "common.h"

namespace maro
{
  namespace datalib
  {
    template <typename T>
    struct ItemContainer_trait
    {
      typedef T type;
    };

    /*
    Binary reader will always return same container for all items,
    user should make sure to copy the return if need
    */
    class ItemContainer
    {
      char* _buffer{ nullptr };
      int _offset{ 0 };

    public:
      ItemContainer();
      ItemContainer(ItemContainer&& cntr) = delete;
      ItemContainer(const ItemContainer& writer) = delete;

      ~ItemContainer();

      /// <summary>
      ///  Set the buffer that use to hold binary data
      /// </summary>
      /// <param name="buffer">Buffer to set</param>
      void set_buffer(char* buffer);

      /// <summary>
      /// Set the offset of current item.
      /// </summary>
      /// <param name="offset">Offset of current item.</param>
      void set_offset(UINT offset);

      /// <summary>
      /// Get value by specified offset, usuall from meta.fields.
      /// </summary>
      /// <typeparam name="T">Type of the field.</typeparam>
      /// <param name="offset">Offset of the field.</param>
      /// <returns>Value of the field.</returns>
      template <typename T>
      typename ItemContainer_trait<T>::type get(int offset);
    };
  } // namespace datalib

} // namespace maro

#endif
