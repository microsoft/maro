# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


# Generate code for raw backend attribute accessors.
raw_backend_path = "maro/backends"

attr_type_list = [
    ("ATTR_CHAR", "Char"),
    ("ATTR_UCHAR", "UChar"),
    ("ATTR_SHORT", "Short"),
    ("ATTR_USHORT", "UShort"),
    ("ATTR_INT", "Int"),
    ("ATTR_UINT", "UInt"),
    ("ATTR_LONG", "Long"),
    ("ATTR_ULONG", "ULong"),
    ("ATTR_FLOAT", "Float"),
    ("ATTR_DOUBLE", "Double"),
]

# Load template for attribute accessors.
attr_acc_template = open(f"{raw_backend_path}/_raw_backend_attr_acc_.pyx.tml").read()

# Base code of raw backend.
raw_backend_code = open(f"{raw_backend_path}/_raw_backend_.pyx").read()

# Real file we use to build.
with open(f"{raw_backend_path}/raw_backend.pyx", "w+") as fp:
    fp.write(raw_backend_code)

    # Append attribute accessor implementations to the end.
    for attr_type_pair in attr_type_list:
        attr_acc_def = attr_acc_template.format(T=attr_type_pair[0], CLSNAME=attr_type_pair[1])

        fp.write(attr_acc_def)
