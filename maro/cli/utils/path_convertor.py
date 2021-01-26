# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


class PathConvertor:
    @staticmethod
    def build_path_without_trailing_slash(path: str) -> str:
        """Build source path without trailing '/'.

        Args:
            path (str): Original path.

        Returns:
            str: Reformatted path.
        """
        if path.endswith("/"):
            path = path[:-1]
        return path

    @staticmethod
    def build_path_with_trailing_slash(path: str) -> str:
        """Build reformatted target dir with trailing '/'.

        Args:
            path: (str): Original path.

        Returns:
            str: Reformatted path.
        """
        if not path.endswith("/"):
            path = path + "/"
        return path
