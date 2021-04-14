// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef  _MARO_BACKENDS_RAW_NODE_
#define _MARO_BACKENDS_RAW_NODE_

#include <vector>
#include <string>
#include <iostream>

#include "common.h"
#include "attribute.h"
#include "bitset.h"

using namespace std;

namespace maro
{
  namespace backends
  {
    namespace raw
    {
      /// <summary>
      /// Extract attribute index from attribute type.
      /// </summary>
      /// <param name="attr_type">Type of attribute.</param>
      /// <returns>Index of this attribute in its node.</returns>
      inline USHORT extract_attr_index(ATTR_TYPE attr_type);

      /// <summary>
      /// Compose the attribute offset in memory block.
      /// </summary>
      /// <param name="node_index">Index of node instance.</param>
      /// <param name="node_size">Per node size in related memory block.</param>
      /// <param name="attr_offset">Attribute offset in node instance.</param>
      /// <param name="slot">Slot index of attribute.</param>
      /// <returns>Attribute offset in memory block.</returns>
      inline size_t compose_attr_offset_in_node(NODE_INDEX node_index, size_t node_size, size_t attr_offset, SLOT_INDEX slot = 0);

      /// <summary>
      /// Definition of attribute.
      /// </summary>
      struct AttributeDef
      {
        // Is this a list attribute.
        bool is_list;

        // Is this a const attribute.
        bool is_const;

        // Data type of this attribute.
        AttrDataType data_type;

        // Number of slot, for fixed size only, list attribute'slot number samed in attribute class.
        SLOT_INDEX slot_number;

        // Type of this attribute.
        ATTR_TYPE attr_type;

        // Offset in each node instance, used to retrieve attribute from node instance.
        size_t offset;

        // Name of attribute.
        string name;

        AttributeDef(string name, AttrDataType data_type, SLOT_INDEX slot_number, size_t offset, bool is_list, bool is_const, ATTR_TYPE attr_type);
      };

      /// <summary>
      /// Node type in memory, there is not node instance in physical, just a concept.
      /// </summary>
      class Node
      {
        friend class SnapshotList;
        friend class Frame;

      private:
        // Memory block to hold dyanmic attributes, these attributes will be copied into snapshot list.
        vector<Attribute> _dynamic_block;

        // Memory block to hold const attribute, these attributes will not be copied into snapshot list,
        // and its value can be set only one time.
        vector<Attribute> _const_block;

        // Attribute defintions of this node type.
        vector<AttributeDef> _attribute_definitions;

        // Used to store all the list of list attribute in this node.
        vector<vector<Attribute>> _list_store;

        // Used to mark which node instance is alive.
        Bitset _node_instance_masks;

        // Number of node instance that alive.
        NODE_INDEX _alive_node_number = 0;

        // Max number of node instance we have (include deleted nodes), used to for padding in snapshot list.
        NODE_INDEX _max_node_number = 0;

        // Node number from definition time, used for reset.
        NODE_INDEX _defined_node_number = 0;

        // Size of each node instance (by attribute number).
        size_t _const_size_per_node = 0;
        size_t _dynamic_size_per_node = 0;

        // Type of this node.
        NODE_TYPE _type = 0;

        // Name of this node type.
        string _name;

        // Is this node been setup.
        bool _is_setup = false;

        // Copy content from source node, for taking snapshot.
        void copy_from(const Node& node, bool is_deep_copy = false);

        // Make sure setup called.
        inline void ensure_setup() const;

        // Make sure attribute index correct.
        inline void ensure_attr_index(USHORT attr_index) const;

        // Make sure node index correct.
        inline void ensure_node_index(NODE_INDEX node_index) const;

        // Get list attribute reference.
        inline Attribute& get_list_attribute(NODE_INDEX node_index, ATTR_TYPE attr_type);

        // Get actual list of a list attribute
        inline vector<Attribute>& get_attribute_list(Attribute& attribute);
      public:
        Node();

        Node(const Node& node);

        Node& operator=(const Node& node);

        /// <summary>
        /// Set type of this node.
        /// </summary>
        /// <param name="type">Type of this node.</param>
        void set_type(NODE_TYPE type) noexcept;

        /// <summary>
        /// Get type of this node.
        /// </summary>
        /// <returns>Type of this node.</returns>
        NODE_TYPE get_type() const noexcept;

        /// <summary>
        /// Set name of this node.
        /// </summary>
        /// <param name="name">Name to set.</param>
        void set_name(string name) noexcept;

        /// <summary>
        /// Get name of this node.
        /// </summary>
        /// <returns>Name of this node.</returns>
        string get_name() const noexcept;

        /// <summary>
        /// Set defined node number, this is the orign node number, used to reset.
        /// </summary>
        /// <param name="number">Number of node instance.</param>
        void set_defined_number(NODE_INDEX number);

        /// <summary>
        /// Get predefined node instance number.
        /// </summary>
        /// <returns>Number of node instance.</returns>
        NODE_INDEX get_defined_number() const noexcept;

        /// <summary>
        /// Get current max node instance number.
        /// </summary>
        /// <returns>Number of max node instance for this node type.</returns>
        NODE_INDEX get_max_number() const noexcept;

        /// <summary>
        /// Get attribute definition.
        /// </summary>
        /// <param name="attr_type">Type of attribute.</param>
        /// <returns>Definition of specified attribute.</returns>
        const AttributeDef& get_attr_definition(ATTR_TYPE attr_type) const;

        /// <summary>
        /// Check if specified node instance is alive.
        /// </summary>
        /// <param name="node_index">Index of specified node instance.</param>
        /// <returns>True if node instance is alive, or false.</returns>
        bool is_node_alive(NODE_INDEX node_index) const noexcept;

        /// <summary>
        /// Get slot number of specified attribute.
        /// </summary>
        /// <param name="node_index">Index of node instance to query.</param>
        /// <param name="attr_type">Type of attribute.</param>
        /// <returns>Slot number of specified attribute, this is the predefined one for normal attributes,
        /// and current list size for list attributes,
        /// </returns>
        SLOT_INDEX get_slot_number(NODE_INDEX node_index, ATTR_TYPE attr_type) const;

        /// <summary>
        /// Initial this node.
        /// </summary>
        void setup();

        /// <summary>
        /// Reset this node to intial state.
        /// </summary>
        void reset();

        /// <summary>
        /// Append node instance for this node type.
        /// </summary>
        /// <param name="node_number">Number of new instance.</param>
        void append_nodes(NODE_INDEX node_number);

        /// <summary>
        /// Remove a node instance.
        /// NOTE: this will not delete the attributes from memory, just mark them as deleted.
        /// </summary>
        /// <param name="node_index">Index of node instance to remove.</param>
        void remove_node(NODE_INDEX node_index);

        /// <summary>
        /// Resume a node instance.
        /// </summary>
        /// <param name="node_index">Index of node instance to resume.</param>
        void resume_node(NODE_INDEX node_index);

        /// <summary>
        /// Add an attribute to this node type.
        /// </summary>
        /// <param name="attr_name">Name of new attribute.</param>
        /// <param name="data_type">Data type of new attribute.</param>
        /// <param name="slot_number">Number of slot for new attribute.</param>
        /// <param name="is_const">Is a const attribute?</param>
        /// <param name="is_list">Is a list attribute?</param>
        /// <returns>Type of new attribute.</returns>
        ATTR_TYPE add_attr(string attr_name, AttrDataType data_type, SLOT_INDEX slot_number = 1, bool is_const = false, bool is_list = false);

        /// <summary>
        /// Get specified attribute from an node instance.
        /// NOTE: this function only used for current frame, not for snapshot, as nodes in snapshot list do not contains
        /// attribute definition.
        /// </summary>
        /// <param name="node_index">Index of node instance.</param>
        /// <param name="attr_type">Type of attribute.</param>
        /// <param name="slot_index">Slot index to query.</param>
        /// <returns>Specified attribute instance.</returns>
        Attribute& get_attr(NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX slot_index);

        /// <summary>
        /// Append a value to list attribute.
        /// </summary>
        /// <typeparam name="T">Data type</typeparam>
        /// <param name="node_index">Index of node instance to append.</param>
        /// <param name="attr_type">Type of attribute.</param>
        /// <param name="value">Value to append</param>
        template<typename T>
        void append_to_list(NODE_INDEX node_index, ATTR_TYPE attr_type, T value);

        /// <summary>
        /// Clear values in a list attribute.
        /// </summary>
        /// <param name="node_index">Index of node instance.</param>
        /// <param name="attr_type">Type of attribute.</param>
        void clear_list(NODE_INDEX node_index, ATTR_TYPE attr_type);

        /// <summary>
        /// Resize size of a list attribute.
        /// </summary>
        /// <param name="node_index">Index of node instance to resize.</param>
        /// <param name="attr_type">Type of attribute.</param>
        /// <param name="new_size">New size to resize.</param>
        void resize_list(NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX new_size);

        /// <summary>
        /// Remove an index from list attribute.
        /// </summary>
        /// <param name="node_index">Index of node instance to resize.</param>
        /// <param name="attr_type">Type of attribute.</param>
        /// <param name="slot_index">Slot index to remove.</param>
        void remove_from_list(NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX slot_index);

        /// <summary>
        /// Insert a value to specified slot.
        /// </summary>
        /// <param name="node_index">Index of node instance to resize.</param>
        /// <param name="attr_type">Type of attribute.</param>
        template<typename T>
        void insert_to_list(NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX slot_index, T value);
      };

      struct OperationsBeforeSetupError : public exception
      {
        const char* what() const noexcept override;
      };

      struct InvalidAttributeDescError : public exception
      {
        const char* what() const noexcept override;
      };

      struct InvalidNodeIndexError : public exception
      {
        const char* what() const noexcept override;
      };

      struct InvalidSlotIndexError : public exception
      {
        const char* what() const noexcept override;
      };

      struct InvalidNodeNumberError : public exception
      {
        const char* what() const noexcept override;
      };

      struct InvalidAttributeTypeError : public exception
      {
        const char* what() const noexcept override;
      };

      struct OperationsAfterSetupError : public exception
      {
        const char* what() const noexcept override;
      };

      struct OperationsOnNonListAttributeError : public exception
      {
        const char* what() const noexcept override;
      };

      struct MaxSlotNumberError : public exception
      {
        const char* what() const noexcept override;
      };
    }
  }
}

#endif // ! _MARO_BACKENDS_RAW_NODE_
