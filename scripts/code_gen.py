


# Generate code for raw backend attribute accessors.
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

attr_acc_template = open("maro/backends/_raw_backend_attr_acc_.pyx.tml").read()

raw_backend_code = open("maro/backends/_raw_backend_.pyx").read()

with open("maro/backends/raw_backend.pyx", "w+") as fp:
    fp.write(raw_backend_code)

    for attr_type_pair in attr_type_list:
        attr_acc_def = attr_acc_template.format(T=attr_type_pair[0], CLSNAME=attr_type_pair[1])

        fp.write(attr_acc_def)