def get_reformatted_source_path(path: str):
    """Build source path without trailing '/'.

    Args:
        path (str): original path.

    Returns:
        str: reformatted path.
    """
    if path.endswith("/"):
        path = path[:-1]
    return path


def get_reformatted_target_dir(path: str):
    """Get reformatted target dir with trailing '/'.

    Args:
        path: (str): original path.

    Returns:
        str: reformatted path.
    """
    if not path.endswith("/"):
        path = path + "/"
    return path
