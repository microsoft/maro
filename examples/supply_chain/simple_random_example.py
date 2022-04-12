from typing import Dict, List

import numpy as np

from maro.simulator import Env
from maro.simulator.scenarios.supply_chain import ConsumerAction, ConsumerUnit, ManufactureAction, ManufactureUnit
from maro.simulator.scenarios.supply_chain.business_engine import SupplyChainBusinessEngine
from maro.simulator.scenarios.supply_chain.facilities import FacilityInfo

CONSUMER_LOWER_BOUND, CONSUMER_UPPER_BOUND = 100, 300
MANUFACTURER_LOWER_BOUND, MANUFACTURER_UPPER_BOUND = 300, 500


def get_metrics(env_: Env) -> dict:
    info = {}
    total_sold = env_.snapshot_list["seller"][env_.tick::"total_sold"].reshape(-1)
    total_demand = env_.snapshot_list["seller"][env_.tick::"total_demand"].reshape(-1)
    info["sold"] = total_sold
    info["demand"] = total_demand
    info["sold/demand"] = info["sold"] / info["demand"]
    return info


if __name__ == '__main__':
    # Create an environment instance
    env = Env(scenario="supply_chain", topology="sample", start_tick=0, durations=100)
    business_engine = env.business_engine
    assert isinstance(business_engine, SupplyChainBusinessEngine)

    # Get all consumers & manufactures in the environment
    entity_list = business_engine.get_entity_list()
    consumers = [entity for entity in entity_list if issubclass(entity.class_type, ConsumerUnit)]
    manufacturers = [entity for entity in entity_list if issubclass(entity.class_type, ManufactureUnit)]

    # Generate the consumer-source mapping. The key of the dictionary is the ID of the consumer unit, while the value
    # of the dictionary is the ID of the upstream facility.
    consumer2source: Dict[int, List[int]] = {}
    facility_info_dict: Dict[int, FacilityInfo] = env.summary["node_mapping"]["facilities"]
    for facility_info in facility_info_dict.values():
        products = facility_info.products_info
        for product_id, product_info in products.items():
            consumer_info = product_info.consumer_info
            if consumer_info is not None:
                consumer2source[consumer_info.id] = consumer_info.source_facility_id_list

    # Initialize the environment with a `None` action
    _, _, is_done = env.step(action=None)
    while not is_done:
        actions = []

        # Generate random actions for all manufacturers
        for entity in manufacturers:
            actions.append(ManufactureAction(
                id=entity.id,
                production_rate=np.random.randint(low=MANUFACTURER_LOWER_BOUND, high=MANUFACTURER_UPPER_BOUND) + 1,
            ))

        # Generate random actions for all consumers that have at least one source
        for entity in consumers:
            if len(consumer2source[entity.id]) > 0:
                actions.append(ConsumerAction(
                    id=entity.id,
                    product_id=entity.skus.id,
                    source_id=np.random.choice(consumer2source[entity.id]),  # Pick a random source
                    quantity=np.random.randint(low=CONSUMER_LOWER_BOUND, high=CONSUMER_UPPER_BOUND) + 1,
                    vehicle_type="train",
                ))
        _, _, is_done = env.step(action=actions)

    # Output the metrics
    print(get_metrics(env))
