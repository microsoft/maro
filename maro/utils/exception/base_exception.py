# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.utils.exception import ERROR_CODE


class MAROException(Exception):
    """ 
    Base exception class for MARO errors in Python.
    
    Args:
    error_code (int): the MARO error code defined in error code.
        e.g. 1000-1999: for errors of the MARO communication toolkits.
    msg (str): Description of the error, if None, show the base error information.
    """

    def __init__(self, error_code: int = 1000, msg: str = None):
        self.error_code = error_code
        self.strerror = msg if msg else ERROR_CODE[self.error_code]

    def __str__(self):
        return self.strerror if isinstance(self.strerror, str) else str(self.strerror)

    def __repr__(self):
        return f"{self.__class__.__name__}({str(self)})"
