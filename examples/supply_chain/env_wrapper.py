# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
from collections import defaultdict

import numpy as np

from maro.rl import AbsEnvWrapper
from maro.simulator.scenarios.supply_chain.actions import ConsumerAction, ManufactureAction


class SCEnvWrapper(AbsEnvWrapper):
    manufacturer_features = [
        "id", "facility_id", "manufacturing_number", "production_rate", "product_id", "storage_id",
        "product_unit_cost", "storage_data_model_index"
    ]

    consumer_features = [
        "id", "facility_id", "product_id", "order_cost", "total_purchased", "total_received", "source_id",
        "quantity", "vlt", "purchased", "received", "order_product_cost"
    ]

    storage_features = ["id", "facility_id", "capacity", "remaining_space", "unit_storage_cost"]

    # Only keep non-ID features in states
    manufacturer_state_indexes = [2, 3, 6]
    consumer_state_indexes = [3, 4, 5, 7, 8, 9, 10, 11]

    def get_state(self, event):
        self.state_info = {}
        # manufacturer state shaping
        manufacturer_snapshots = self.env.snapshot_list["manufacture"]
        storage_snapshots = self.env.snapshot_list["storage"]
        manufacturer_features = manufacturer_snapshots[self.env.frame_index::SCEnvWrapper.manufacturer_features]
        manufacturer_features = manufacturer_features.flatten().reshape(len(manufacturer_snapshots), -1).astype(np.int32)
        
        # combine manufacture state and the corresponding storage state
        state = {
            feature[0]: np.concatenate([
                feature[SCEnvWrapper.manufacturer_state_indexes],
                storage_snapshots[self.env.frame_index:feature[-1]:SCEnvWrapper.storage_features[2:]].flatten()
            ]).astype(np.float32)
            for feature in manufacturer_features
        }

        # consumer state shaping
        consumer_snapshots = self.env.snapshot_list["consumer"]
        consumer_features = consumer_snapshots[self.env.frame_index::SCEnvWrapper.consumer_features]
        consumer_features = consumer_features.flatten().reshape(len(consumer_snapshots), -1).astype(np.int32)
        state.update({
            feature[0]: np.asarray(feature[SCEnvWrapper.consumer_state_indexes], dtype=np.float32)
            for feature in consumer_features
        })
        self.state_info = {feature[0]: {"product_id": feature[2]} for feature in consumer_features}

        return state

    def get_action(self, action_by_agent):
        # cache the sources for each consumer if not yet cached
        if not hasattr(self, "consumer_source"):
            self.consumer_source = {}
            for id_, (type_, idx) in self.env.summary["node_mapping"]["unit_mapping"].items():
                if type_ == "consumer":
                    sources = self.env.snapshot_list["consumer"][self.env.frame_index:idx:"sources"]
                    if sources:
                        sources = sources.flatten().astype(np.int)
                        self.consumer_source[id_] = sources

        env_action = {}
        for agent_id, action in action_by_agent.items():
            # consumer action
            if agent_id in self.state_info:
                source_id = random.choice(self.consumer_source[agent_id]) if agent_id in self.consumer_source else 0
                product_id = self.state_info[agent_id]["product_id"]
                env_action[agent_id] = ConsumerAction(agent_id, product_id, source_id, action, 1)
            # manufacturer action
            else:
                env_action[agent_id] = ManufactureAction(agent_id, action)

        return env_action

    def get_reward(self, tick=None):
        return np.float32(np.random.rand())
