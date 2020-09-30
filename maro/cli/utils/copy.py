# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


def get_reformatted_source_path(path: str):
    """Build source path without trailing '/'.

    Args:
        path (str): Original path.

    Returns:
        str: Reformatted path.
    """
    if path.endswith("/"):
        path = path[:-1]
    return path


def get_reformatted_target_dir(path: str):
    """Get reformatted target dir with trailing '/'.

    Args:
        path: (str): Original path.

    Returns:
        str: Reformatted path.
    """
    if not path.endswith("/"):
        path = path + "/"
    return path
