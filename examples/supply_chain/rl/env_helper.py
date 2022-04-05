# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from typing import Any, Dict

from maro.simulator import Env
from maro.simulator.scenarios.supply_chain.business_engine import SupplyChainBusinessEngine
from maro.simulator.scenarios.supply_chain.facilities import FacilityInfo
from maro.simulator.scenarios.supply_chain.objects import SupplyChainEntity
from maro.simulator.scenarios.supply_chain.units import (
    ConsumerUnitInfo, DistributionUnitInfo, ManufactureUnitInfo, ProductUnitInfo, SellerUnitInfo, StorageUnitInfo
)

from .config import env_conf


# Create an env to extract some required information
env = Env(**env_conf)

# agent naming
helper_business_engine = env.business_engine
assert isinstance(helper_business_engine, SupplyChainBusinessEngine)
entity_dict: Dict[Any, SupplyChainEntity] = {entity.id: entity for entity in helper_business_engine.get_entity_list()}

# storage info
num_skus = len(env.summary["node_mapping"]["skus"]) + 1  # TODO: why + 1?
STORAGE_INFO = {
    "facility_levels": {},
    "unit2facility": {},
    "facility2storage": {},   # facility id -> storage index
    "storage_product_num": {},  # facility id -> product id -> number
    "storage_product_indexes": defaultdict(dict),  # facility id -> product_id -> index
    "facility_product_utilization": {},  # facility id -> storage product utilization
    # use this to quick find relationship between units (consumer, manufacture, seller or product) and product unit.
    # unit id  -> (product unit id, facility id, seller id, consumer id, manufacture id)
    "unit2product": {},
}

# facility levels
facility_info_dict: Dict[int, FacilityInfo] = env.summary["node_mapping"]["facilities"]

for facility_id, facility_info in facility_info_dict.items():

    STORAGE_INFO["facility_levels"][facility_id] = {
        "node_index": facility_info.node_index,
        "config": facility_info.configs,
        "upstreams": facility_info.upstreams,
        "skus": facility_info.skus,
    }

    storage: StorageUnitInfo = facility_info.storage_info
    if storage is not None:
        STORAGE_INFO["facility_levels"][facility_id]["storage"] = storage
        STORAGE_INFO["unit2facility"][storage.id] = facility_id
        STORAGE_INFO["facility2storage"][facility_id] = storage.node_index
        STORAGE_INFO["storage_product_num"][facility_id] = [0] * num_skus
        STORAGE_INFO["facility_product_utilization"][facility_id] = 0

        for i, pid in enumerate(storage.product_list):
            STORAGE_INFO["storage_product_indexes"][facility_id][pid] = i
            STORAGE_INFO["storage_product_num"][facility_id][pid] = 0

    distribution: DistributionUnitInfo = facility_info.distribution_info
    if distribution is not None:
        STORAGE_INFO["facility_levels"][facility_id]["distribution"] = distribution
        STORAGE_INFO["unit2facility"][distribution.id] = facility_id

    products: Dict[int, ProductUnitInfo] = facility_info.products_info
    if products:
        for product_id, product in products.items():
            product_info = {
                "skuproduct": product
            }
            STORAGE_INFO["unit2facility"][product.id] = facility_id
            seller: SellerUnitInfo = product.seller_info
            if seller is not None:
                product_info["seller"] = seller
                STORAGE_INFO["unit2facility"][seller.id] = facility_id
            consumer: ConsumerUnitInfo = product.consumer_info
            if consumer is not None:
                product_info["consumer"] = consumer
                STORAGE_INFO["unit2facility"][consumer.id] = facility_id
            manufacture: ManufactureUnitInfo = product.manufacture_info
            if manufacture is not None:
                product_info["manufacture"] = manufacture
                STORAGE_INFO["unit2facility"][manufacture.id] = facility_id

            STORAGE_INFO["facility_levels"][facility_id][product_id] = product_info

            for unit in (seller, consumer, manufacture, product):
                if unit is not None:
                    STORAGE_INFO["unit2product"][unit.id] = (
                        product.id,
                        facility_id,
                        seller.id if seller is not None else None,
                        consumer.id if consumer is not None else None,
                        manufacture.id if manufacture is not None else None,
                    )
