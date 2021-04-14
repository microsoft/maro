from maro.simulator import Env
from collections import defaultdict, namedtuple
import scipy.stats as st
import numpy as np
from maro.rl import AbsEnvWrapper
from maro.simulator.scenarios.supply_chain.actions import ConsumerAction, ManufactureAction


def stock_constraint(f_state):
    return 0 < f_state['inventory_in_stock'] <= (f_state['max_vlt'] + 7) * f_state['sale_mean']


def is_replenish_constraint(f_state):
    return f_state['consumption_hist'][-1] > 0


def low_profit(f_state):
    return (f_state['sku_price'] - f_state['sku_cost']) * f_state['sale_mean'] <= 1000


def low_stock_constraint(f_state):
    return 0 < f_state['inventory_in_stock'] <= (f_state['max_vlt'] + 3) * f_state['sale_mean']


def out_of_stock(f_state):
    return 0 < f_state['inventory_in_stock']


atoms = {
    'stock_constraint': stock_constraint,
    'is_replenish_constraint': is_replenish_constraint,
    'low_profit': low_profit,
    'low_stock_constraint': low_stock_constraint,
    'out_of_stock': out_of_stock
}

# State extracted.
keys_in_state = [(None, ['is_over_stock', 'is_out_of_stock', 'is_below_rop', 
                         'constraint_idx', 'is_accepted', 'consumption_hist']),
                 ('storage_capacity', ['storage_utilization']),
                 ('sale_gamma', ['sale_std',
                                 'sale_hist',
                                 'pending_order',
                                 'inventory_in_stock',
                                 'inventory_in_transit',
                                 'inventory_estimated',
                                 'inventory_rop']),
                 ('max_price', ['sku_price', 'sku_cost'])]


class UnitBaseInfo:
    id: int = None
    node_index: int = None
    config: dict = None
    summary: dict = None

    def __init__(self, unit_summary):
        self.id = unit_summary["id"]
        self.node_index = unit_summary["node_index"]
        self.config = unit_summary.get("config", {})
        self.summary = unit_summary

    def __getitem__(self, key, default=None):
        if key in self.summary:
            return self.summary[key]

        return default


distribution_features = ("remaining_order_quantity", "remaining_order_number")
seller_features = ("total_demand", "sold", "demand")


class SCEnvWrapper(AbsEnvWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.balance_cal = BalanceSheetCalculator(env)
        self.cur_balance_sheet_reward = None
        self.storage_ss = env.snapshot_list["storage"]
        self.distribution_ss = env.snapshot_list["distribution"]
        self.consumer_ss = env.snapshot_list["consumer"]
        self.seller_ss = env.snapshot_list["seller"]

        self._summary = env.summary['node_mapping']
        self._configs = env.configs
        self._agent_types = self._summary["agent_types"]
        self._units_mapping = self._summary["unit_mapping"]
        self._agent_list = env.agent_idx_list

        self._sku_number = len(self._summary["skus"]) + 1
        self._max_price = self._summary["max_price"]
        self._max_sources_per_facility = self._summary["max_sources_per_facility"]

        # state for each tick
        self._cur_metrics = env.metrics

        # cache for ppf value.
        self._service_index_ppf_cache = {}

        # facility -> {
        # data_model_index:int,
        # storage:UnitBaseInfo,
        # distribution: UnitBaseInfo,
        # product_id: {
        # consumer: UnitBaseInfo,
        # seller: UnitBaseInfo,
        # manufacture: UnitBaseInfo
        # }
        # }
        self.facility_levels = {}

        # unit id -> (facility id)
        self.unit_2_facility_dict = {}

        # our raw state
        self._states = {}

        # facility id -> storage index
        self._facility2storage_index_dict = {}

        # facility id -> product id -> number
        self._storage_product_numbers = {}

        # facility id -> product_id -> index
        self._storage_product_indices = {}

        # facility id -> storage product utilization
        self._facility_product_utilization = {}

        # facility id -> in_transit_orders
        self._facility_in_transit_orders = {}

        # current distribution states
        self._cur_distribution_states = None

        # current consumer states
        self._cur_consumer_states = None

        # current seller states
        self._cur_seller_states = None

        # dim for state
        self._dim = None

        # built internal helpers.
        self._build_internal_helpers()

    @property
    def dim(self):
        """Calculate dim per shape."""
        if self._dim is None:
            self._dim = 0

            first_state = next(iter(self._states.values()))

            for _, state_keys in keys_in_state:
                for key in state_keys:
                    val = first_state[key]

                    if type(val) == list:
                        self._dim += len(val)
                    else:
                        self._dim += 1

        return self._dim

    def get_state(self, event):
        cur_tick = self.env.tick
        settings: dict = self.env.configs.settings
        consumption_hist_len = settings['consumption_hist_len']
        hist_len = settings['sale_hist_len']
        consumption_ticks = [cur_tick -
                             i for i in range(consumption_hist_len-1, -1, -1)]
        hist_ticks = [cur_tick - i for i in range(hist_len-1, -1, -1)]

        self.cur_balance_sheet_reward = self.balance_cal.calc()
        self._cur_metrics = self.env.metrics

        self._cur_distribution_states = self.distribution_ss[cur_tick::distribution_features].flatten(
        ).reshape(-1, 2).astype(np.int)
        self._cur_consumer_states = self.consumer_ss[consumption_ticks::"latest_consumptions"].flatten(
        ).reshape(-1, len(self.consumer_ss))
        self._cur_seller_states = self.seller_ss[hist_ticks::seller_features].astype(
            np.int)

        # facility level states
        for facility_id in self._facility_product_utilization:
            # reset for each step
            self._facility_product_utilization[facility_id] = 0

            in_transit_orders = self._cur_metrics['facilities'][facility_id]["in_transit_orders"]

            self._facility_in_transit_orders[facility_id] = [
                0] * self._sku_number

            for sku_id, number in in_transit_orders.items():
                self._facility_in_transit_orders[facility_id][sku_id] = number

        final_state = {}

        # calculate storage info first, then use it later to speed up.
        for facility_id, storage_index in self._facility2storage_index_dict.items():
            product_numbers = self.storage_ss[cur_tick:storage_index:"product_number"].flatten(
            ).astype(np.int)

            for pid, index in self._storage_product_indices[facility_id].items():
                product_number = product_numbers[index]

                self._storage_product_numbers[facility_id][pid] = product_number
                self._facility_product_utilization[facility_id] += product_number

        for agent_info in self._agent_list:
            state = self._states[agent_info.id]

            storage_index = self._facility2storage_index_dict[agent_info.facility_id]

            self._update_facility_features(state, agent_info)
            self._update_storage_features(state, agent_info)
            # bom do not need to update
            # self._add_bom_features(state, agent_info)
            self._update_distribution_features(state, agent_info)
            self._update_sale_features(state, agent_info)
            # vlt do not need to update
            # self._update_vlt_features(state, agent_info)
            self._update_consumer_features(state, agent_info)
            # self._add_price_features(state, agent_info)
            self._update_global_features(state)

            np_state = self._serialize_state(state)

            final_state[f"consumer.{agent_info.id}"] = np_state
            final_state[f"producer.{agent_info.id}"] = np_state

        return final_state

    def get_reward(self, tick=None, target_agents=None):
        wc = self.env.configs.settings["global_reward_weight_consumer"]
        parent_facility_balance = {}
        for f_id, sheet in self.cur_balance_sheet_reward.items():
            if f_id in self.unit_2_facility_dict:
                # it is a product unit
                parent_facility_balance[f_id] = self.cur_balance_sheet_reward[self.unit_2_facility_dict[f_id]]
            else:
                parent_facility_balance[f_id] = sheet

        consumer_reward_by_facility = {f_id: wc * parent_facility_balance[f_id][0] + (1 - wc) * bsw[1] for f_id, bsw in
                                       self.cur_balance_sheet_reward.items()}

        return {
            **{f"producer.{f_id}": np.float32(reward[0]) for f_id, reward in self.cur_balance_sheet_reward.items()},
            **{f"consumer.{f_id}": np.float32(reward) for f_id, reward in consumer_reward_by_facility.items()}
        }

    def get_action(self, action_by_agent):
        # cache the sources for each consumer if not yet cached
        if not hasattr(self, "product2source"):
            self.product2source, self.consumer2product = {}, {}
            for facility in self.env.summary["node_mapping"]["facilities"].values():
                products = facility["units"]["products"]
                for product_id, product in products.items():
                    consumer = product["consumer"]
                    if consumer is not None:
                        consumer_id = consumer["id"]
                        product_unit_id = product["id"]
                        self.product2source[product_unit_id] = consumer["sources"]
                        self.consumer2product[consumer_id] = product_id

        env_action = {}
        for agent_id, action in action_by_agent.items():
            unit_id = int(agent_id.split(".")[1])

            is_facility = unit_id not in self._units_mapping

            # ignore facility to reduce action number
            if is_facility:
                continue

            # consumer action
            if agent_id.startswith("consumer"):
                product_id = self.consumer2product.get(unit_id, 0)
                sources = self.product2source.get(unit_id, [])

                if sources:
                    source_id = sources[0]

                    action_number = int(int(action) * self._cur_metrics["products"][unit_id]["sale_mean"])

                    # ignore 0 quantity to reduce action number
                    if action_number == 0:
                        continue

                    sku = self._units_mapping[unit_id][3]
                    reward_discount = 0

                    env_action[unit_id] = ConsumerAction(unit_id, product_id, source_id, action_number, sku.vlt, reward_discount)
            # manufacturer action
            elif agent_id.startswith("producer"):
                sku = self._units_mapping[unit_id][3]
                action = sku.production_rate

                # ignore invalid actions
                if action is None or action == 0:
                    continue

                env_action[unit_id] = ManufactureAction(unit_id, action)

        return env_action

    def _update_facility_features(self, state, agent_info):
        state['is_positive_balance'] = 1 if self.balance_cal.total_balance_sheet[agent_info.id] > 0 else 0

    def _update_storage_features(self, state, agent_info):
        facility_id = agent_info.facility_id
        state['storage_utilization'] = 0

        state['storage_levels'] = self._storage_product_numbers[facility_id]
        state['storage_utilization'] = self._facility_product_utilization[facility_id]

    def _update_sale_features(self, state, agent_info):
        if agent_info.is_facility:
            return

        product_metrics = self._cur_metrics["products"][agent_info.id]

        # for product unit only
        state['sale_mean'] = product_metrics["sale_mean"]
        state['sale_std'] = product_metrics["sale_std"]

        facility = self.facility_levels[agent_info.facility_id]
        product_info = facility[agent_info.sku.id]

        if "seller" not in product_info:
            # TODO: why gamma sale as mean?
            state['sale_gamma'] = state['sale_mean']

        if "consumer" in product_info:
            consumer_index = product_info["consumer"].node_index

            state['consumption_hist'] = list(
                self._cur_consumer_states[:, consumer_index])
            state['pending_order'] = list(
                product_metrics["pending_order_daily"])

        if "seller" in product_info:
            seller_index = product_info["seller"].node_index

            seller_states = self._cur_seller_states[:, seller_index, :]

            # for total demand, we need latest one.
            state['total_backlog_demand'] = seller_states[:, 0][-1][0]
            state['sale_hist'] = list(seller_states[:, 1].flatten())
            state['backlog_demand_hist'] = list(seller_states[:, 2])

    def _update_distribution_features(self, state, agent_info):
        facility = self.facility_levels[agent_info.facility_id]
        distribution = facility.get("distribution", None)

        if distribution is not None:
            dist_states = self._cur_distribution_states[distribution.node_index]
            state['distributor_in_transit_orders'] = dist_states[1]
            state['distributor_in_transit_orders_qty'] = dist_states[0]

    def _update_consumer_features(self, state, agent_info):
        if agent_info.is_facility:
            return

        facility = self.facility_levels[agent_info.facility_id]
        product_info = facility[agent_info.sku.id]

        if "consumer" not in product_info:
            return

        state['consumer_in_transit_orders'] = self._facility_in_transit_orders[agent_info.facility_id]

        product_index = self._storage_product_indices[agent_info.facility_id][agent_info.sku.id]
        state['inventory_in_stock'] = self._storage_product_numbers[agent_info.facility_id][product_index]
        state['inventory_in_transit'] = state['consumer_in_transit_orders'][agent_info.sku.id]

        pending_order = self._cur_metrics["facilities"][agent_info.facility_id]["pending_order"]

        if pending_order is not None:
            state['inventory_in_distribution'] = pending_order[agent_info.sku.id]

        state['inventory_estimated'] = (state['inventory_in_stock']
                                        + state['inventory_in_transit']
                                        - state['inventory_in_distribution'])
        if state['inventory_estimated'] >= 0.5 * state['storage_capacity']:
            state['is_over_stock'] = 1

        if state['inventory_estimated'] <= 0:
            state['is_out_of_stock'] = 1

        service_index = state['service_level']

        if service_index not in self._service_index_ppf_cache:
            self._service_index_ppf_cache[service_index] = st.norm.ppf(
                service_index)

        ppf = self._service_index_ppf_cache[service_index]

        state['inventory_rop'] = (state['max_vlt'] * state['sale_mean']
                                  + np.sqrt(state['max_vlt']) * state['sale_std'] * ppf)

        if state['inventory_estimated'] < state['inventory_rop']:
            state['is_below_rop'] = 1

    def _update_global_features(self, state):
        state["global_time"] = self.env.tick

    def _serialize_state(self, state):
        result = []

        for norm, fields in keys_in_state:
            for field in fields:
                vals = state[field]
                if not isinstance(vals, list):
                    vals = [vals]
                if norm is not None:
                    vals = [max(0.0, min(100.0, x / (state[norm] + 0.01)))
                            for x in vals]
                result.extend(vals)

        return np.asarray(result, dtype=np.float32)

    def _build_internal_helpers(self):
        # facility levels
        for facility_id, facility in self._summary["facilities"].items():
            self.facility_levels[facility_id] = {
                "node_index": facility["node_index"],
                "config": facility['configs'],
                "upstreams": facility["upstreams"],
                "skus": facility["skus"]
            }

            units = facility["units"]

            storage = units["storage"]
            if storage is not None:
                self.facility_levels[facility_id]["storage"] = UnitBaseInfo(
                    storage)

                self.unit_2_facility_dict[storage["id"]] = facility_id

                self._facility2storage_index_dict[facility_id] = storage["node_index"]

                self._storage_product_numbers[facility_id] = [
                    0] * self._sku_number
                self._storage_product_indices[facility_id] = {}
                self._facility_product_utilization[facility_id] = 0

                for i, pid in enumerate(storage["product_list"]):
                    self._storage_product_indices[facility_id][pid] = i
                    self._storage_product_numbers[facility_id][pid] = 0

            distribution = units["distribution"]

            if distribution is not None:
                self.facility_levels[facility_id]["distribution"] = UnitBaseInfo(
                    distribution)
                self.unit_2_facility_dict[distribution["id"]] = facility_id

            products = units["products"]

            if products:
                for product_id, product in products.items():
                    product_info = {
                        "skuproduct": UnitBaseInfo(product)
                    }

                    self.unit_2_facility_dict[product["id"]] = facility_id

                    seller = product['seller']

                    if seller is not None:
                        product_info["seller"] = UnitBaseInfo(seller)
                        self.unit_2_facility_dict[seller["id"]] = facility_id

                    consumer = product["consumer"]

                    if consumer is not None:
                        product_info["consumer"] = UnitBaseInfo(consumer)
                        self.unit_2_facility_dict[consumer["id"]] = facility_id

                    manufacture = product["manufacture"]

                    if manufacture is not None:
                        product_info["manufacture"] = UnitBaseInfo(manufacture)
                        self.unit_2_facility_dict[manufacture["id"]
                                                  ] = facility_id

                    self.facility_levels[facility_id][product_id] = product_info

        # create initial state structure
        self._build_init_state()

    def _build_init_state(self):
        # we will build the final state with default and const values,
        # then update dynamic part per step
        for agent_info in self._agent_list:
            state = {}

            facility = self.facility_levels[agent_info.facility_id]

            # global features
            state["global_time"] = 0

            # facility features
            state["facility"] = None
            state["facility_type"] = [1 if i == agent_info.agent_type else 0 for i in range(len(self._agent_types))]
            state["is_accepted"] = [0] * self._configs.settings["constraint_state_hist_len"]
            state['constraint_idx'] = [0]
            state['facility_id'] = [0] * self._sku_number
            state['sku_info'] = {} if agent_info.is_facility else agent_info.sku
            state['echelon_level'] = 0

            state['facility_info'] = facility['config']
            state["is_positive_balance"] = 0

            if not agent_info.is_facility:
                state['facility_id'][agent_info.sku.id] = 1

            for atom_name in atoms.keys():
                state[atom_name] = list(
                    np.ones(self._configs.settings['constraint_state_hist_len']))

            # storage features
            state['storage_levels'] = [0] * self._sku_number
            state['storage_capacity'] = facility['storage'].config["capacity"]
            state['storage_utilization'] = 0

            # bom features
            state['bom_inputs'] = [0] * self._sku_number
            state['bom_outputs'] = [0] * self._sku_number

            if not agent_info.is_facility:
                state['bom_inputs'][agent_info.sku.id] = 1
                state['bom_outputs'][agent_info.sku.id] = 1

            # vlt features
            sku_list = self._summary["skus"]
            current_source_list = []

            if agent_info.sku is not None:
                current_source_list = facility["upstreams"].get(
                    agent_info.sku.id, [])

            state['vlt'] = [0] * \
                (self._max_sources_per_facility * self._sku_number)
            state['max_vlt'] = 0

            if not agent_info.is_facility:
                # only for sku product
                product_info = facility[agent_info.sku.id]

                if "consumer" in product_info and len(current_source_list) > 0:
                    state['max_vlt'] = product_info["skuproduct"]["max_vlt"]

                    for i, source in enumerate(current_source_list):
                        for j, sku in enumerate(sku_list.values()):
                            # NOTE: different with original code, our config can make sure that source has product we need

                            if sku.id == agent_info.sku.id:
                                state['vlt'][i * len(sku_list) + j +
                                             1] = facility["skus"][sku.id].vlt

            # sale features
            settings = self.env.configs.settings
            hist_len = settings['sale_hist_len']
            consumption_hist_len = settings['consumption_hist_len']

            state['sale_mean'] = 1.0
            state['sale_std'] = 1.0
            state['sale_gamma'] = 1.0
            state['service_level'] = 0.95
            state['total_backlog_demand'] = 0

            state['sale_hist'] = [0] * hist_len
            state['backlog_demand_hist'] = [0] * hist_len
            state['consumption_hist'] = [0] * consumption_hist_len
            state['pending_order'] = [0] * settings['pending_order_len']

            if not agent_info.is_facility:
                state['service_level'] = agent_info.sku.service_level

                product_info = facility[agent_info.sku.id]

                if "seller" in product_info:
                    state['sale_gamma'] = facility["skus"][agent_info.sku.id].sale_gamma

            # distribution features
            state['distributor_in_transit_orders'] = 0
            state['distributor_in_transit_orders_qty'] = 0

            # consumer features
            state['consumer_source_export_mask'] = [0] * \
                (self._max_sources_per_facility * self._sku_number)
            state['consumer_source_inventory'] = [0] * self._sku_number
            state['consumer_in_transit_orders'] = [0] * self._sku_number

            state['inventory_in_stock'] = 0
            state['inventory_in_transit'] = 0
            state['inventory_in_distribution'] = 0
            state['inventory_estimated'] = 0
            state['inventory_rop'] = 0
            state['is_over_stock'] = 0
            state['is_out_of_stock'] = 0
            state['is_below_rop'] = 0

            if len(current_source_list) > 0:
                for i, source in enumerate(current_source_list):
                    for j, sku in enumerate(sku_list.values()):
                        if sku.id == agent_info.sku.id:
                            state['consumer_source_export_mask'][i * len(sku_list) + j + 1] = \
                                self.facility_levels[source]["skus"][sku.id].vlt

            # price features
            state['max_price'] = self._max_price
            state['sku_price'] = 0
            state['sku_cost'] = 0

            if not agent_info.is_facility:
                state['sku_price'] = agent_info.sku.price
                state['sku_cost'] = agent_info.sku.cost

            self._states[agent_info.id] = state


class BalanceSheetCalculator:
    consumer_features = ("id", "order_quantity", "price",
                         "order_cost", "order_product_cost", "reward_discount")
    seller_features = ("id", "sold", "demand", "price", "backlog_ratio")
    manufacture_features = ("id", "manufacturing_number", "product_unit_cost")
    product_features = (
        "id", "price", "distribution_check_order", "distribution_transport_cost", "distribution_delay_order_penalty")
    storage_features = ("capacity", "remaining_space")
    vehicle_features = ("id", "payload", "unit_transport_cost")

    def __init__(self, env: Env):
        self.env = env
        self.consumer_ss = env.snapshot_list["consumer"]
        self.seller_ss = env.snapshot_list["seller"]
        self.manufacture_ss = env.snapshot_list["manufacture"]
        self.storage_ss = env.snapshot_list["storage"]
        self.distribution_ss = env.snapshot_list["distribution"]
        self.vehicle_ss = env.snapshot_list["vehicle"]
        self.product_ss = env.snapshot_list["product"]
        self.products = []
        self.product_id2index_dict = {}
        self.facility_levels = []

        self.facilities = env.summary["node_mapping"]["facilities"]

        for facility_id, facility in self.facilities.items():
            pid_list = []
            distribution = facility["units"]["distribution"]

            for product_id, product in facility["units"]["products"].items():
                pid_list.append(product["id"])
                consumer = product["consumer"]
                seller = product["seller"]
                manufacture = product["manufacture"]

                self.product_id2index_dict[product["id"]] = len(self.products)

                self.products.append((
                    product["id"],
                    product_id,
                    facility["units"]["storage"]["node_index"],
                    facility["units"]["storage"]["config"]["unit_storage_cost"],
                    distribution["node_index"] if distribution is not None else None,
                    facility["downstreams"],
                    None if consumer is None else (
                        consumer["id"], consumer["node_index"]),
                    None if seller is None else (
                        seller["id"], seller["node_index"]),
                    None if manufacture is None else (
                        manufacture["id"], manufacture["node_index"]),
                ))

            self.facility_levels.append((
                facility_id,
                pid_list,
                facility["units"]["storage"]["node_index"],
                facility["units"]["storage"]["config"]["unit_storage_cost"],
                distribution["node_index"] if distribution is not None else None,
                [v["node_index"] for v in distribution["children"]
                 ] if distribution is not None else []
            ))

        self.total_balance_sheet = defaultdict(int)

    def calc(self):
        tick = self.env.tick
        # consumer
        consumer_bs_states = self.consumer_ss[tick::self.consumer_features].flatten().reshape(-1, len(
            self.consumer_features))

        # quantity * price
        consumer_profit = consumer_bs_states[:, 1] * consumer_bs_states[:, 2]

        # balance_sheet_profit = 0
        # order_cost + order_product_cost
        consumer_step_balance_sheet_loss = -1 * \
            (consumer_bs_states[:, 3] + consumer_bs_states[:, 4])

        # consumer step reward: balance sheet los + profile * discount
        consumer_step_reward = consumer_step_balance_sheet_loss + \
            consumer_profit * consumer_bs_states[:, 5]

        # seller
        seller_bs_states = self.seller_ss[tick::self.seller_features].flatten(
        ).reshape(-1, len(self.seller_features))

        # profit = sold * price
        seller_balance_sheet_profit = seller_bs_states[:,
                                                       1] * seller_bs_states[:, 3]

        # loss = demand * price * backlog_ratio
        seller_balance_sheet_loss = -1 * \
            seller_bs_states[:, 2] * \
            seller_bs_states[:, 3] * seller_bs_states[:, 4]

        # step reward = loss + profit
        seller_step_reward = seller_balance_sheet_loss + seller_balance_sheet_profit

        # manufacture
        man_bs_states = self.manufacture_ss[tick::self.manufacture_features].flatten().reshape(-1, len(
            self.manufacture_features))

        # loss = manufacture number * cost
        man_balance_sheet_profit_loss = -1 * \
            man_bs_states[:, 1] * man_bs_states[:, 2]

        # step reward = loss
        man_step_reward = man_balance_sheet_profit_loss

        # product
        product_bs_states = self.product_ss[tick::self.product_features].flatten().reshape(-1,
                                                                                           len(self.product_features))

        # product distribution loss = check order + delay order penalty
        product_distribution_balance_sheet_loss = -1 * \
            (product_bs_states[:, 3] + product_bs_states[:, 4])

        # product distribution profit = check order * price
        product_distribution_balance_sheet_profit = product_bs_states[:,
                                                                      2] * product_bs_states[:, 1]

        # result we need
        product_step_reward = np.zeros((len(self.products, )))
        product_balance_sheet_profit = np.zeros((len(self.products, )))
        product_balance_sheet_loss = np.zeros((len(self.products, )))

        # create product number mapping for storages
        storages_product_map = {}
        for storage_index in range(len(self.storage_ss)):
            product_list = self.storage_ss[tick:storage_index:"product_list"].flatten(
            ).astype(np.int)
            product_number = self.storage_ss[tick:storage_index:"product_number"].flatten(
            ).astype(np.int)

            storages_product_map[storage_index] = {
                pid: pnum for pid, pnum in zip(product_list, product_number)}

        # product balance sheet and reward
        # loss = consumer loss + seller loss + manufacture loss + storage loss + distribution loss + downstreams loss
        # profit = same as above
        # reward = same as above
        for i, product in enumerate(self.products):
            id, product_id, storage_index, unit_storage_cost, distribution_index, downstreams, consumer, seller, manufacture = product

            if consumer:
                product_balance_sheet_loss[i] += consumer_step_balance_sheet_loss[consumer[1]]
                product_step_reward[i] += consumer_step_reward[consumer[1]]

            if seller:
                product_balance_sheet_loss[i] += seller_balance_sheet_loss[seller[1]]
                product_balance_sheet_profit[i] += seller_balance_sheet_profit[seller[1]]
                product_step_reward[i] += seller_step_reward[seller[1]]

            if manufacture:
                product_balance_sheet_loss[i] += man_balance_sheet_profit_loss[manufacture[1]]
                product_step_reward[i] += man_step_reward[manufacture[1]]

            storage_reward = -1 * \
                storages_product_map[storage_index][product_id] * \
                unit_storage_cost

            product_step_reward[i] += storage_reward

            product_balance_sheet_loss[i] += storage_reward

            if distribution_index is not None:
                product_balance_sheet_loss[i] += product_distribution_balance_sheet_loss[distribution_index]
                product_balance_sheet_profit[i] += product_distribution_balance_sheet_profit[distribution_index]

                product_step_reward[i] += product_distribution_balance_sheet_loss[distribution_index] + \
                    product_distribution_balance_sheet_profit[distribution_index]

            if downstreams and len(downstreams) > 0:
                if product_id in downstreams:
                    for dfacility in downstreams[product_id]:
                        dproducts = self.facilities[dfacility]["units"]["products"]

                        did = dproducts[product_id]["id"]

                        product_balance_sheet_loss[i] += product_balance_sheet_loss[self.product_id2index_dict[did]]
                        product_balance_sheet_profit[i] += product_balance_sheet_profit[self.product_id2index_dict[did]]
                        product_step_reward[i] += product_step_reward[self.product_id2index_dict[did]]

        product_balance_sheet = product_balance_sheet_profit + product_balance_sheet_loss

        # storage
        storage_states = self.storage_ss[tick::self.storage_features].flatten(
        ).reshape(-1, len(self.storage_features))

        # loss = (capacity-remaining space) * cost
        storage_balance_sheet_loss = -1 * \
            (storage_states[:, 0] - storage_states[:, 1])

        # vehicles
        vehicle_states = self.vehicle_ss[tick::self.vehicle_features].flatten(
        ).reshape(-1, len(self.vehicle_features))

        # loss = cost * payload
        vehicle_balance_sheet_loss = -1 * \
            vehicle_states[:, 1] * vehicle_states[:, 2]
        vehicle_step_reward = vehicle_balance_sheet_loss

        facility_balance_sheet_loss = np.zeros((len(self.facility_levels),))
        facility_balance_sheet_profit = np.zeros((len(self.facility_levels),))
        facility_step_reward = np.zeros((len(self.facility_levels),))

        # for facilities
        for i, facility in enumerate(self.facility_levels):
            id, pid_list, storage_index, unit_storage_cost, distribution_index, vehicle_indices = facility

            # storage balance sheet
            # profit=0
            facility_balance_sheet_loss[i] += storage_balance_sheet_loss[storage_index] * \
                unit_storage_cost

            # distribution balance sheet
            if distribution_index is not None:
                for vindex in vehicle_indices:
                    facility_balance_sheet_loss[i] += vehicle_balance_sheet_loss[vindex]
                    # distribution unit do not provide reward

            # sku product unit balance sheet
            for pid in pid_list:
                facility_balance_sheet_loss[i] += product_balance_sheet_loss[self.product_id2index_dict[pid]]
                facility_balance_sheet_profit[i] += product_balance_sheet_profit[self.product_id2index_dict[pid]]
                facility_step_reward[i] += product_step_reward[self.product_id2index_dict[pid]]

        result = {}

        for id, bs, rw in zip([item[0] for item in self.products], product_balance_sheet, product_step_reward):
            result[id] = (bs, rw)

            self.total_balance_sheet[id] += bs

        facility_balance_sheet = facility_balance_sheet_loss + facility_balance_sheet_profit

        for id, bs, rw in zip([item[0] for item in self.facility_levels], facility_balance_sheet, facility_step_reward):
            result[id] = (bs, rw)

            self.total_balance_sheet[id] += bs

        return result


if __name__ == "__main__":
    from time import time
    import cProfile

    env = Env(
        scenario="supply_chain",
        topology="random",
        durations=100,
        max_snapshots=10)

    ss = SCEnvWrapper(env)

    env.step(None)

    start_time = time()

    # cProfile.run("ss.get_state(None)", sort="cumtime")
    states = ss.get_state(None)

    end_time = time()

    print("time cost:", end_time - start_time)

    print("dim:", ss.dim)
