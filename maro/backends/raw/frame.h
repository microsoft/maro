// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef  _MARO_BACKENDS_RAW_FRAME_
#define _MARO_BACKENDS_RAW_FRAME_

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "common.h"
#include "attribute.h"
#include "node.h"

using namespace std;

namespace maro
{
  namespace backends
  {
    namespace raw
    {
      /// <summary>
      /// Extract node type from attribute type.
      /// </summary>
      /// <param name="attr_type">Type of attribute.</param>
      /// <returns>Type of node.</returns>
      inline NODE_TYPE extract_node_type(ATTR_TYPE attr_type);

      /// <summary>
      /// A frame used to hold nodes and their attribute, it can be a current frame or a snapshot in snapshot list.
      /// </summary>
      class Frame
      {
        friend class SnapshotList;

      private:
        // All node types, index is the NODE_TYPE.
        vector<Node> _nodes;

        // Is current frame instance already being set up.
        bool _is_setup = false;

        // Copy from another frame, used for taking snapshot.
        inline void copy_from(const Frame& frame);

        // Make sure frame already setup.
        inline void ensure_setup();

        // Make sure node type correct.
        inline void ensure_node_type(NODE_TYPE node_type);

      public:
        Frame();

        /// <summary>
        /// Copy contents from another frame, deep copy.
        /// </summary>
        /// <param name="frame">Source frame to copy.</param>
        Frame(const Frame& frame);

        /// <summary>
        /// Copy contents from another frame, for taking snapshot,
        /// copy without name, const block and attribute definitions.
        /// </summary>
        /// <param name="frame">Source frame to copy.</param>
        /// <returns>Current frame instance.</returns>
        Frame& operator=(const Frame& frame);

        /// <summary>
        /// Add a node type in frame.
        /// </summary>
        /// <param name="node_name">Name of the new node type.</param>
        /// <param name="node_number">Number of initial instance for this node type.</param>
        /// <returns>Node type used to identify this kind of node.</returns>
        NODE_TYPE add_node(string node_name, NODE_INDEX node_number);

        /// <summary>
        /// Add an attribute for specified node type.
        /// </summary>
        /// <param name="node_type">Type of node.</param>
        /// <param name="attr_name">Name of new attribute.</param>
        /// <param name="data_type">Data type of new attribute, default is int.</param>
        /// <param name="slot_number">How many slot of this attribute, default is 1.</param>
        /// <param name="is_const">Is this is a const attribute?</param>
        /// <param name="is_list">Is this a list attribute that without fixed slot number.</param>
        /// <returns>Type of this attribute.</returns>
        ATTR_TYPE add_attr(NODE_TYPE node_type, string attr_name,
          AttrDataType data_type = AttrDataType::AINT, SLOT_INDEX slot_number = 1,
          bool is_const = false, bool is_list = false);

        /// <summary>
        /// Get specified node.
        /// </summary>
        /// <param name="node_type">Type of node.</param>
        /// <returns>Target node reference.</returns>
        Node& get_node(NODE_TYPE node_type);

        /// <summary>
        /// Add node instance for specified node type.
        /// </summary>
        /// <param name="node_type">Type of node.</param>
        /// <param name="node_number">Number to append.</param>
        void append_node(NODE_TYPE node_type, NODE_INDEX node_number);

        /// <summary>
        /// Remove specified node instace from node type.
        /// </summary>
        /// <param name="node_type">Type of node.</param>
        /// <param name="node_index">Index of node instance to remove.</param>
        void remove_node(NODE_TYPE node_type, NODE_INDEX node_index);

        /// <summary>
        /// Resume a node instance.
        /// </summary>
        /// <param name="node_type">Type of node.</param>
        /// <param name="node_index">Index of node instance to resume.</param>
        void resume_node(NODE_TYPE node_type, NODE_INDEX node_index);

        /// <summary>
        /// Get value from specified attribute.
        /// </summary>
        /// <typeparam name="T">Type of attribute value.</typeparam>
        /// <param name="node_index">Index of the node instance.</param>
        /// <param name="attr_type">Type of the attribute.</param>
        /// <param name="slot_index">Which slot to query.</param>
        /// <returns>Value of attribute.</returns>
        template<typename T>
        typename Attribute_Trait<T>::type get_value(NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX slot_index);

        /// <summary>
        /// Set value for specified attribute.
        /// </summary>
        /// <typeparam name="T">Type of attribute.</typeparam>
        /// <param name="node_index">Index of node instance to set.</param>
        /// <param name="attr_type">Type of attribute.</param>
        /// <param name="slot_index">Which slot to set.</param>
        /// <param name="value">Value to set.</param>
        template<typename T>
        void set_value(NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX slot_index, T value);

        /// <summary>
        /// Append a value to a list attribute.
        /// </summary>
        /// <typeparam name="T">Type of the value.</typeparam>
        /// <param name="node_index">Index of node instance to set.</param>
        /// <param name="attr_type">Type of attribute.</param>
        /// <param name="value">Value to append.</param>
        template<typename T>
        void append_to_list(NODE_INDEX node_index, ATTR_TYPE attr_type, T value);

        /// <summary>
        /// Clear a list attribute.
        /// </summary>
        /// <param name="node_index">Index of node instance to clear.</param>
        /// <param name="attr_type">Type of attribute to clear</param>
        void clear_list(NODE_INDEX node_index, ATTR_TYPE attr_type);

        /// <summary>
        /// Resize a list attribute with specified size.
        /// </summary>
        /// <param name="node_index">Index of node instance to resize.</param>
        /// <param name="attr_type">Type of attribute.</param>
        /// <param name="new_size">New size to resize.</param>
        void resize_list(NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX new_size);

        /// <summary>
        /// Remove specified slot from list attribute.
        /// </summary>
        /// <param name="node_index">Index of node instance to resize.</param>
        /// <param name="attr_type">Type of attribute.</param>
        /// <param name="slot_index">Slot to remove.</param>
        void remove_from_list(NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX slot_index);

        /// <summary>
        /// Insert a value to specified slot for list attribute.
        /// </summary>
        /// <param name="node_index">Index of node instance to resize.</param>
        /// <param name="attr_type">Type of attribute.</param>
        /// <param name="slot_index">Slot to insert.</param>
        /// <param name="value">Value to insert. </param>
        template<typename T>
        void insert_to_list(NODE_INDEX node_index, ATTR_TYPE attr_type, SLOT_INDEX slot_index, T value);

        /// <summary>
        /// Initial current frame.
        /// </summary>
        void setup();

        /// <summary>
        /// Reset current frame, it will recover the node instance number to pre-defined one.
        /// </summary>
        void reset();

        /// <summary>
        /// Dump current frame content into specified folder, nodes will be dump into
        /// different files.
        /// </summary>
        /// <param name="folder">Folder to dump file.</param>
        void dump(string folder);

        /// <summary>
        /// Check if specified node type exist or not.
        /// </summary>
        /// <param name="node_type">Type of node</param>
        /// <returns>True if exist, or false.</returns>
        bool is_node_exist(NODE_TYPE node_type) const noexcept;

        SLOT_INDEX get_slot_number(NODE_INDEX node_index, ATTR_TYPE attr_type);

      };


      /// <summary>
      /// Operations before frame being setup.
      /// </summary>
      struct FrameNotSetupError : public exception
      {
        const char* what() const noexcept override;
      };


      /// <summary>
      /// Try to add new node/attribute type after seting up.
      /// </summary>
      struct FrameAlreadySetupError : public exception
      {
        const char* what() const noexcept override;
      };


      /// <summary>
      /// Invalid node type.
      /// </summary>
      struct FrameBadNodeTypeError : public exception
      {
        const char* what() const noexcept override;
      };


      /// <summary>
      /// Invalid attribute type.
      /// </summary>
      struct FrameBadAttributeTypeError : public exception
      {
        const char* what() const noexcept override;
      };

      struct FrameInvalidNodeNumerError : public exception
      {
        const char* what() const noexcept override;
      };
    }
  }
}

#endif // ! _MARO_BACKENDS_RAW_FRAME_
