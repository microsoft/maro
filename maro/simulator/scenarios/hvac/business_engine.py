# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from yaml import safe_load

from maro.backends.frame import FrameBase, SnapshotList
from maro.event_buffer import AtomEvent, CascadeEvent, EventBuffer, MaroEvents
from maro.simulator.scenarios import AbsBusinessEngine
from maro.simulator.scenarios.helpers import DocableDict
from maro.utils.utils import convert_dottable

from .ahu import AHU
from .ahu_prediction import ahu_pred_model
from .common import Action, PendingDecisionPayload
from .events import Events
from .event_payload import AHUSetPayload
from .frame_builder import gen_hvac_frame

metrics_desc = """
HVAC: Heating, Ventilation, Air-conditioning and Cooling.
AHU: Air Handling Unit.

The HVAC metrics used provide statistics information until now.
It contains following keys:

total_ahu_kw (float): The total energy consumption for all the AHU until now (unit: kW).
"""


class HvacBusinessEngine(AbsBusinessEngine):
    def __init__(
        self, event_buffer: EventBuffer, topology: str, start_tick: int,
        max_tick: int, snapshot_resolution: int, max_snapshots: int,
        additional_options: dict={}
    ):
        super().__init__(
            "hvac", event_buffer, topology, start_tick, max_tick,
            snapshot_resolution, max_snapshots, additional_options
        )

        self._load_configs()
        self._register_events()

        self._init_frame()
        self._ahus: List[AHU] = self._frame.ahus
        self._ahu_mat_path_list, self._ahu_predictor = self._init_ahus()

        self._init_data_reader()

        self._init_metrics()

    def _load_configs(self):
        # Update self._config_path with current file path.
        self.update_config_root_path(__file__)

        with open(os.path.join(self._config_path, "config.yml")) as fp:
            self._config = convert_dottable(safe_load(fp))

    def _init_frame(self):
        self._frame = gen_hvac_frame(
            snapshots_num=self.calc_max_snapshots(),
            ahu_num=len(self._config.ahu)
        )
        self._snapshots = self._frame.snapshots

    def _init_ahus(self) -> List[str]:
        ahu_mat_path_list: List[str] = []
        ahu_predictor: List[Tuple[nn.Module, MinMaxScaler, MinMaxScaler]] = []

        for ahu, ahu_setting in zip(self._ahus, self._config.ahu):
            ahu.set_init_state(
                name=ahu_setting.name,
                mat=ahu_setting.initial_values.mat,
                dat=ahu_setting.initial_values.dat,
                at=ahu_setting.initial_values.at,
                kw=ahu_setting.initial_values.kw,
                sps=ahu_setting.initial_values.sps,
                das=ahu_setting.initial_values.das
            )
            ahu_mat_path_list.append(ahu_setting.mat_path)

            predictor = ahu_pred_model()
            predictor.load_state_dict(torch.load(ahu_setting.transition.paths.model))
            predictor.eval()

            x_scaler = joblib.load(ahu_setting.transition.paths.x_scaler)
            y_scaler = joblib.load(ahu_setting.transition.paths.y_scaler)

            ahu_predictor.append((predictor, x_scaler, y_scaler))

        return ahu_mat_path_list, ahu_predictor

    def _init_data_reader(self):
        # TODO: replaced with binary reader
        def read_in_mat_list(mat_path):
            df = pd.read_csv(mat_path, sep=',', delimiter=None, header='infer')
            df = df.dropna()
            df = df.reset_index()
            return df['MAT']

        self._mat_list = [read_in_mat_list(mat_path) for mat_path in self._ahu_mat_path_list]

        ########################################################################
        data_path = "/home/Jinyu/maro/maro/simulator/scenarios/hvac/topologies/building121/datasets/train_data_AHU_MAT.csv"
        df = pd.read_csv(data_path, sep=',', delimiter=None, header='infer')
        df = df.dropna()
        df = df.reset_index()

        baseline = {
            "kw": df["KW"].to_numpy(),
            "dat": df["DAT"].to_numpy(),
            "at": df["air_ton"].to_numpy(),
            "mat": df["DAS"].to_numpy() + df["delta_MAT_DAS"].to_numpy(),
            "sps": df["SPS"].to_numpy(),
            "das": df["DAS"].to_numpy(),
            "total_kw": np.cumsum(df["KW"].to_numpy())
        }

        self._statistics = {
            key: {
                "mean": np.mean(baseline[key]),
                "min": np.min(baseline[key]),
                "max": np.max(baseline[key]),
                "range": np.max(baseline[key]) - np.min(baseline[key]),
            }
            for key in baseline.keys()
        }

    def _init_metrics(self):
        self._total_ahu_kw: float = 0

    def _register_events(self):
        register_handler = self._event_buffer.register_event_handler

        register_handler(event_type=MaroEvents.TAKE_ACTION, handler=self._on_action_received)
        # Use the fake one now
        register_handler(event_type=Events.AHU_SET, handler=self._on_ahu_set)

    def _on_ahu_set(self, event: AtomEvent):
        payload: Action = event.payload
        ahu: AHU = self._ahus[payload.ahu_idx]
        ahu.sps = payload.sps
        ahu.das = payload.das

        predictor, x_scaler, y_scaler = self._ahu_predictor[ahu.idx]

        x = np.array([ahu.sps, ahu.das, ahu.mat - ahu.das]).reshape(1, -1)
        x = torch.tensor(x_scaler.transform(x))
        y_pred = predictor(x).detach().numpy()
        y_pred = y_scaler.inverse_transform(y_pred)[0]

        # ahu.kw, ahu.at, ahu.dat = y_pred

        ########################################################################
        ahu.kw = max(y_pred[0], self._statistics["kw"]["min"])
        ahu.at = max(y_pred[1], self._statistics["at"]["min"])
        ahu.dat = max(y_pred[2], self._statistics["dat"]["min"])

    def _on_action_received(self, event: CascadeEvent):
        for action in event.payload:
            # Set SPS and DAS at next tick to align with the logic used by Bonsai
            ahu_set_payload = action
            ahu_set_event = self._event_buffer.gen_atom_event(tick=event.tick + 1, event_type=Events.AHU_SET, payload=ahu_set_payload)
            self._event_buffer.insert_event(event=ahu_set_event)

    @property
    def frame(self) -> FrameBase:
        """FrameBase: Frame instance of current business engine."""
        return self._frame

    @property
    def snapshots(self) -> SnapshotList:
        """SnapshotList: Snapshot list of current frame, this is used to expose querying interface for outside."""
        return self._snapshots

    @property
    def configs(self) -> dict:
        """dict: Configurations of this business engine."""
        return self._config

    def step(self, tick: int):
        """Method that is called at each tick, usually used to trigger business logic at current tick.

        Args:
            tick (int): Current tick from simulator.
        """
        if tick > 0:
            for ahu in self._ahus:
                # TODO: to align with the logic used for Bonsai, tick - 1 as index here
                ahu.mat = self._mat_list[ahu.idx][tick-1]

        for ahu in self._ahus:
            pending_decision_payload = PendingDecisionPayload(ahu_idx=ahu.idx)
            pending_decision_event = self._event_buffer.gen_decision_event(tick=tick, payload=pending_decision_payload)
            self._event_buffer.insert_event(pending_decision_event)

    def post_step(self, tick: int) -> bool:
        """This method will be called at the end of each tick, used to post-process for each tick,
        for complex business logic with many events, it maybe not easy to determine
        if stop the scenario at the middle of tick, so this method is used to avoid this.

        Args:
            tick (int): Current tick.

        Returns:
            bool: If simulator should stop simulation at current tick.
        """
        for ahu in self._ahus:
            self._total_ahu_kw += ahu.kw

        if (tick + 1) % self._snapshot_resolution == 0:
            self._frame.take_snapshot(self.frame_index(tick))

        return tick + 1 == self._max_tick

    def reset(self):
        """Reset states business engine."""
        self._snapshots.reset()
        self._frame.reset()

        for ahu in self._ahus:
            ahu.reset()

        self._init_metrics()

    def get_event_payload_detail(self) -> dict:
        """Get payload keys for all kinds of event.

        For the performance of the simulator, some event payload has no corresponding Python object.
        This mapping is provided for your convenience in such case.

        Returns:
            dict: Key is the event type in string format, value is a list of available keys.
        """
        return {
            Events.AHU_SET.name: AHUSetPayload.summary_key,
            Events.PENDING_DECISION.name: PendingDecisionPayload.summary_key,
        }

    def get_metrics(self) -> DocableDict:
        """Get statistics information, may different for scenarios.

        Returns:
            dict: Dictionary about metrics, content and format determined by business engine.
        """
        return DocableDict(
            doc=metrics_desc,
            total_ahu_kw=self._total_ahu_kw
        )
