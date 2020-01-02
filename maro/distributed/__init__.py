# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.distributed.message import Message
from maro.distributed.proxy import Proxy
from maro.distributed.dist_decorator import dist


__all__ = ['dist', 'Proxy', 'Message']
