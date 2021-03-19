# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


ERROR_CODE = {
    # Error code table for the MARO system.
    # 1000-1999: Error code for the communication module.
    1000: "MARO Internal Error",
    1001: "Redis Connection Error",
    1002: "Proxy Peers Missing Error",
    1003: "Redis Information Uncompleted Error",
    1004: "Peers Connection Error",
    1005: "Driver Send Error",
    1006: "Driver Receive Error",
    1007: "Message Session Type Error",
    1008: "Conditional Event Syntax Error",
    1009: "Driver Type Error",
    1010: "Socket Type Error",
    1011: "Peers Disconnection Error",
    1012: "MARO Send Again Error",
    1013: "Peers Rejoin Timeout",

    # data lib
    2000: "Meta does not contain timestamp field",
    2001: "Invalid vessel parking duration time, it must large than 0, please check",

    # backends
    2100: "Invalid parameter to get attribute value, please use tuple, list or slice instead",
    2101: "Invalid parameter to set attribute value",
    2102: "Cannot set value for frame fields directly if slot number more than 1, please use slice interface instead",
    2103: "Append method only support for list attribute.",
    2104: "Resize method only support for list attribute.",
    2105: "Clear method only fupport for list attribute.",
    2106: "Insert method only fupport for list attribute.",
    2107: "Remove method only fupport for list attribute.",
    2108: "Node already been deleted.",
    2109: "Node not exist.",
    2110: "Invalid attribute.",

    # simulator
    2200: "Cannot find specified business engine",

    # 3000-3999: Error code for CLI
    3000: "CLI Internal Error",
    3001: "Command Error",
    3002: "Parsing Error",
    3003: "Deployment Error",

    # 4000-4999: Error codes for RL toolkit
    4000: "Store Misalignment",
    4001: "Missing Optimizer",
    4002: "Unrecognized Task",
}
