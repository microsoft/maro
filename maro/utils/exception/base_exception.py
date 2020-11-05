# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from .error_code import ERROR_CODE


class MAROException(Exception):
    """The base exception class for MARO.

    Args:
        error_code (int): the predefined MARO error code. You can find the
            detailed definition in: `maro.utils.exception.error_code.py`.
        msg (str): Description of the error. Defaults to None, which will
            show the base error information.
    """

    def __init__(self, error_code: int = 1000, msg: str = None):
        self.error_code = error_code
        self.strerror = msg if msg else ERROR_CODE[self.error_code]

    def __str__(self):
        return self.strerror if isinstance(self.strerror, str) else str(self.strerror)

    def __repr__(self):
        return f"{self.__class__.__name__}({str(self)})"
