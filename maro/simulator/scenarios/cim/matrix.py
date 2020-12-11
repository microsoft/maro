# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.backends.frame import NodeAttribute, NodeBase, node
from maro.simulator.scenarios.matrix_accessor import MatrixAttributeAccessor


def gen_matrix(port_num: int, vessel_num: int):
    """A node that contains matrix in frame.

    Args:
        port_num (int): Number of ports.
        vessel_num (int): Number of vessels.

    Return:
        type: Matrix class definition.
    """
    @node("matrices")
    class GeneralInfoMatrix(NodeBase):
        """Used to save matrix, and provide matrix accessor."""

        # distribution of full from port to port
        full_on_ports = NodeAttribute("i", slot_num=port_num * port_num)
        # distribution of full from vessel to port
        full_on_vessels = NodeAttribute("i", slot_num=vessel_num * port_num)
        # planed route info for vessels
        vessel_plans = NodeAttribute("i", slot_num=vessel_num * port_num)

        def __init__(self):
            # we cannot create matrix accessor here, since the attributes will be bind after frame setup,
            self._acc_dict = {}
            self._acc_dict["full_on_ports"] = MatrixAttributeAccessor(self, "full_on_ports", port_num, port_num)
            self._acc_dict["full_on_vessels"] = MatrixAttributeAccessor(self, "full_on_vessels", vessel_num, port_num)
            self._acc_dict["vessel_plans"] = MatrixAttributeAccessor(self, "vessel_plans", vessel_num, port_num)

        def __getitem__(self, key):
            if key in self._acc_dict:
                return self._acc_dict[key]

            return None

    return GeneralInfoMatrix
