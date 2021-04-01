# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.backends.backend import AttributeType
from maro.backends.frame import NodeAttribute, node

from .skumodel import SkuDataModel


@node("product")
class ProductDataModel(SkuDataModel):

    def __init__(self):
        super(ProductDataModel, self).__init__()

    def reset(self):
        super(ProductDataModel, self).reset()
