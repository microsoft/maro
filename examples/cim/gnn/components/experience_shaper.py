from collections import defaultdict

import numpy as np

from maro.rl import ColumnBasedStore


class ExperienceShaper:
    def __init__(
            self, static_list, dynamic_list, max_tick, gnn_state_shaper, 
            scale_factor=0.0001, 
            time_slot=100,
            discount_factor=0.97
        ):
        self._static_list = list(static_list)
        self._dynamic_list = list(dynamic_list)
        self._time_slot = time_slot
        self._discount_factor = discount_factor
        self._discount_vector = np.logspace(1, self._time_slot, self._time_slot, base=discount_factor)
        self._max_tick = max_tick
        self._tick_range = list(range(self._max_tick))
        self._len_return = self._max_tick - self._time_slot
        self._gnn_state_shaper = gnn_state_shaper
        self._fulfillment_list, self._shortage_list, self._experience_dict = None, None, None
        self._experience_dict = defaultdict(ColumnBasedStore())
        self._init_state()
        self._scale_factor = scale_factor
        self._state_keys = ["v", "p", "vo", "po", "vedge", "pedge"]

    def _init_state(self):
        self._fulfillment_list, self._shortage_list = np.zeros(self._max_tick + 1), np.zeros(self._max_tick + 1)
        self._experience_dict = defaultdict(list)
        self._last_tick = 0

    def record(self, decision_event, model_action, model_input):
        # Only the experience that has the next state of given time slot is valuable.
        if decision_event.tick + self._time_slot < self._max_tick:
            self._experience_dict[decision_event.port_idx, decision_event.vessel_idx].put({
                "tick": decision_event.tick,
                "s": model_input,
                "a": model_action,
            })

    def _compute_delta(self, arr):
        delta = np.array(arr)
        delta[1:] -= arr[:-1]
        return delta

    def _batch_obs_to_numpy(self, obs):
        v = np.stack([o["v"] for o in obs], axis=0)
        p = np.stack([o["p"] for o in obs], axis=0)
        vo = np.stack([o["vo"] for o in obs], axis=0)
        po = np.stack([o["po"] for o in obs], axis=0)
        return {"p": p, "v": v, "vo": vo, "po": po}

    def __call__(self, snapshot_list):
        shortage = snapshot_list["ports"][self._tick_range:self._static_list:"shortage"].reshape(self._max_tick, -1)
        fulfillment = snapshot_list["ports"][self._tick_range:self._static_list:"fulfillment"] \
            .reshape(self._max_tick, -1)
        delta = fulfillment - shortage
        R = np.empty((self._len_return, len(self._static_list)), dtype=np.float)
        for i in range(0, self._len_return, 1):
            R[i] = np.dot(self._discount_vector, delta[i + 1: i + self._time_slot + 1])

        for (agent_idx, vessel_idx), exp_list in self._experience_dict.items():
            for exp in exp_list:
                tick = exp["tick"]
                exp["s_"] = self._gnn_state_shaper(tick=tick + self._time_slot)
                exp["R"] = self._scale_factor * R[tick]

        return {
            agent_id: {
                "s": {
                    key: np.stack([e["s"][key] for e in exp_list], axis=1 if key in {"v", "p"} else 0)
                    for key in self._state_keys
                },
                "s_": {
                    key: np.stack([e["s"][key] for e in exp_list], axis=1 if key in {"v", "p"} else 0)
                    for key in self._state_keys
                },
                "a": np.array([e["a"] for e in exp_list], dtype=np.int64),
                "R": np.vstack([e["R"] for e in exp_list]),
                "len": len(exp_list)
            }
            for agent_id, exp_list in self._experience_dict.items()
        }

    def reset(self):
        del self._experience_dict
        self._init_state()
