from typing import Dict, List, Tuple

import numpy as np

from maro.simulator import Env
from maro.simulator.scenarios.supply_chain import ConsumerAction, ConsumerUnit
from maro.simulator.scenarios.supply_chain.business_engine import SupplyChainBusinessEngine
from maro.simulator.scenarios.supply_chain.facilities import FacilityInfo
from maro.simulator.scenarios.supply_chain.objects import SupplyChainEntity

CONSUMER_LOWER_BOUND, CONSUMER_UPPER_BOUND = 100, 300


def get_metrics(env_: Env) -> dict:
    info = {}
    total_sold = env_.snapshot_list["seller"][env_.tick :: "total_sold"].reshape(-1)
    total_demand = env_.snapshot_list["seller"][env_.tick :: "total_demand"].reshape(-1)
    info["sold"] = total_sold
    info["demand"] = total_demand
    info["sold/demand"] = info["sold"] / info["demand"]
    return info


if __name__ == "__main__":
    # Create an environment instance
    env = Env(scenario="supply_chain", topology="single_echelon", start_tick=0, durations=100)
    business_engine = env.business_engine
    assert isinstance(business_engine, SupplyChainBusinessEngine)

    # Get all consumers in the environment
    entity_list = business_engine.get_entity_list()
    consumers = [entity for entity in entity_list if issubclass(entity.class_type, ConsumerUnit)]
    print(f"\n[Consumer] {len(consumers)} Consumers in total!")
    for consumer in consumers:
        print(f"  Consumer id = {consumer.id}, facility id = {consumer.facility_id}, sku id = {consumer.skus.id}, sku name = {consumer.skus.name}")

    # Generate the consumer-source mapping. The key of the dictionary is the ID of the consumer unit, while the value
    # of the dictionary is the ID of the upstream facility.
    consumer2source: Dict[int, List[int]] = {}
    facility_info_dict: Dict[int, FacilityInfo] = env.summary["node_mapping"]["facilities"]
    for facility_info in facility_info_dict.values():
        products = facility_info.products_info
        for product_info in products.values():
            consumer_info = product_info.consumer_info
            if consumer_info is not None:
                consumer2source[consumer_info.id] = consumer_info.source_facility_id_list

    print(f"\n[Consumer - source facility id list]")
    for consumer_id, facility_id_list in consumer2source.items():
        print(f"  Consumer id = {consumer_id}, source facility id list: {facility_id_list}")

    # Initialize the environment with a `None` action
    _, _, is_done = env.step(action=None)
    while not is_done:
        actions = []

        # Generate random actions for all consumers that have at least one source
        for entity in consumers:
            if len(consumer2source[entity.id]) > 0:
                actions.append(
                    ConsumerAction(
                        id=entity.id,
                        sku_id=entity.skus.id,
                        source_id=np.random.choice(consumer2source[entity.id]),  # Pick a random source
                        quantity=np.random.randint(low=CONSUMER_LOWER_BOUND, high=CONSUMER_UPPER_BOUND) + 1,
                        vehicle_type="train",
                    ),
                )
        _, _, is_done = env.step(action=actions)

    # Output the metrics
    print(get_metrics(env))
