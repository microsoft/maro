# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import random
from typing import Dict, List, Optional

import dgl
import dgl.function as fn
import torch

from maro.backends.frame import FrameBase, SnapshotList
from maro.event_buffer import CascadeEvent, EventBuffer, MaroEvents
from maro.simulator.scenarios import AbsBusinessEngine

from examples.mis.lwd.simulator.common import Action, MISDecisionPayload, MISEnvMetrics, VertexState


class MISBusinessEngine(AbsBusinessEngine):
    def __init__(
        self,
        event_buffer: EventBuffer,
        topology: Optional[str],
        start_tick: int,
        max_tick: int,
        snapshot_resolution: int,
        max_snapshots: Optional[int],
        additional_options: dict = None,
    ) -> None:
        super(MISBusinessEngine, self).__init__(
            scenario_name="MaximumIndependentSet",
            event_buffer=event_buffer,
            topology=topology,
            start_tick=start_tick,
            max_tick=max_tick,
            snapshot_resolution=snapshot_resolution,
            max_snapshots=max_snapshots,
            additional_options=additional_options,
        )
        self._config = self._parse_additional_options(additional_options)

        # NOTE: not used now
        self._frame: FrameBase = FrameBase()
        self._snapshots = self._frame.snapshots

        self._register_events()

        self._batch_graphs: dgl.DGLGraph = None
        self._vertex_states: torch.Tensor = None
        self._is_done_mask: torch.Tensor = None

        self._num_included_record: Dict[int, torch.Tensor] = None
        self._hamming_distance_among_samples_record: Dict[int, torch.Tensor] = None

        self.reset()

    def _parse_additional_options(self, additional_options: dict) -> dict:
        required_keys = [
            "graph_batch_size",
            "num_samples",
            "device",
            "num_node_lower_bound",
            "num_node_upper_bound",
            "node_sample_probability",
        ]
        for key in required_keys:
            assert key in additional_options, f"Parameter {key} is required in additional_options!"

        self._graph_batch_size = additional_options["graph_batch_size"]
        self._num_samples = additional_options["num_samples"]
        self._device = additional_options["device"]
        self._num_node_lower_bound = additional_options["num_node_lower_bound"]
        self._num_node_upper_bound = additional_options["num_node_upper_bound"]
        self._node_sample_probability = additional_options["node_sample_probability"]

        return {key: additional_options[key] for key in required_keys}

    @property
    def configs(self) -> dict:
        return self._config

    @property
    def frame(self) -> FrameBase:
        return self._frame

    @property
    def snapshots(self) -> SnapshotList:
        return self._snapshots

    def _register_events(self) -> None:
        self._event_buffer.register_event_handler(MaroEvents.TAKE_ACTION, self._on_action_received)

    def get_agent_idx_list(self) -> List[int]:
        return [0]

    def set_seed(self, seed: int) -> None:
        pass

    def _calculate_and_record_metrics(self, tick: int, undecided_before_mask: torch.Tensor) -> None:
        # Calculate and record the number of included vertexes for cardinality reward.
        included_mask = self._vertex_states == VertexState.Included
        self._batch_graphs.ndata["h"] = included_mask.float()
        node_count = dgl.sum_nodes(self._batch_graphs, "h")
        self._batch_graphs.ndata.pop("h")
        self._num_included_record[tick] = node_count

        # Calculate and record the diversity for diversity reward.
        if self._num_samples == 2:
            undecided_before_mask_left, undecided_before_mask_right = undecided_before_mask.split(1, dim=1)

            states_left, states_right = self._vertex_states.split(1, dim=1)
            undecided_mask_left = states_left == VertexState.Deferred
            undecided_mask_right = states_right == VertexState.Deferred

            hamming_distance = torch.abs(states_left.float() - states_right.float())
            hamming_distance[undecided_mask_left | undecided_mask_right] = 0.0
            hamming_distance[~undecided_before_mask_left & ~undecided_before_mask_right] = 0.0

            self._batch_graphs.ndata["h"] = hamming_distance
            distance = dgl.sum_nodes(self._batch_graphs, "h").expand_as(node_count)
            self._batch_graphs.ndata.pop("h")
            self._hamming_distance_among_samples_record[tick] = distance

        return

    def _on_action_received(self, event: CascadeEvent) -> None:
        actions = event.payload
        assert isinstance(actions, List)

        undecided_before_mask = self._vertex_states == VertexState.Deferred

        # Update Phase
        for action in actions:
            assert isinstance(action, Action)
            undecided_mask = self._vertex_states == VertexState.Deferred
            self._vertex_states[undecided_mask] = action.vertex_states[undecided_mask]

        # Clean-Up Phase: Set clashed node pairs to Deferred
        included_mask = self._vertex_states == VertexState.Included
        self._batch_graphs.ndata["h"] = included_mask.float()
        self._batch_graphs.update_all(fn.copy_u(u="h", out="m"), fn.sum(msg="m", out="h"))
        neighbor_included_mask = self._batch_graphs.ndata.pop("h").bool()

        # Clashed: if the node and its neighbor are both set to included
        clashed_mask = included_mask & neighbor_included_mask
        self._vertex_states[clashed_mask] = VertexState.Deferred
        neighbor_included_mask[clashed_mask] = False

        # Clean-Up Phase: exclude the deferred vertex neighboring to an included one.
        undecided_mask = self._vertex_states == VertexState.Deferred
        self._vertex_states[undecided_mask & neighbor_included_mask] = VertexState.Excluded

        # Timeout handling
        if event.tick + 1 == self._max_tick:
            undecided_mask = self._vertex_states == VertexState.Deferred
            self._vertex_states[undecided_mask] = VertexState.Excluded

        self._calculate_and_record_metrics(event.tick, undecided_before_mask)
        self._update_is_done_mask()

        return

    def step(self, tick: int) -> None:
        decision_payload = MISDecisionPayload(
            graph=self._batch_graphs,
            vertex_states=self._vertex_states.clone(),
        )
        decision_event = self._event_buffer.gen_decision_event(tick, decision_payload)
        self._event_buffer.insert_event(decision_event)

    def _generate_er_graph(self) -> dgl.DGLGraph:
        num_nodes = random.randint(self._num_node_lower_bound, self._num_node_upper_bound)

        w = -1
        lp = math.log(1.0 - self._node_sample_probability)

        # Nodes in graph are from 0, num_nodes - 1 (start with v as the first node index).
        v = 1
        u_list, v_list = [], []
        while v < num_nodes:
            lr = math.log(1.0 - random.random())
            w = w + 1 + int(lr / lp)
            while w >= v and v < num_nodes:
                w = w - v
                v = v + 1
            if v < num_nodes:
                u_list.extend([v, w])
                v_list.extend([w, v])

        graph = dgl.graph((u_list, v_list), num_nodes=num_nodes)

        return graph

    def reset(self, keep_seed: bool = False) -> None:
        self._batch_graphs = dgl.batch([self._generate_er_graph() for _ in range(self._graph_batch_size)])
        self._batch_graphs.set_n_initializer(dgl.init.zero_initializer)
        self._batch_graphs = self._batch_graphs.to(self._device)

        tensor_size = (self._batch_graphs.num_nodes(), self._num_samples)
        self._vertex_states = torch.full(size=tensor_size, fill_value=VertexState.Deferred, device=self._device)
        self._update_is_done_mask()

        self._num_included_record = {}
        self._hamming_distance_among_samples_record = {}
        return

    def _update_is_done_mask(self) -> None:
        undecided_mask = self._vertex_states == VertexState.Deferred
        self._batch_graphs.ndata["h"] = undecided_mask.float()
        num_undecided = dgl.sum_nodes(self._batch_graphs, "h")
        self._batch_graphs.ndata.pop("h")
        self._is_done_mask = (num_undecided == 0)
        return

    def post_step(self, tick: int) -> bool:
        if tick + 1 == self._max_tick:
            return True

        return torch.all(self._is_done_mask).item()

    def get_metrics(self) -> dict:
        return {
            MISEnvMetrics.IncludedNodeCount: self._num_included_record,
            MISEnvMetrics.HammingDistanceAmongSamples: self._hamming_distance_among_samples_record,
            MISEnvMetrics.IsDoneMasks: self._is_done_mask.cpu().detach().numpy(),
        }


if __name__ == "__main__":
    from maro.simulator import Env
    device = torch.device("cuda:0")

    env = Env(
        business_engine_cls=MISBusinessEngine,
        options={
            "graph_batch_size": 4,
            "num_samples": 2,
            "device": device,
            "num_node_lower_bound": 15,
            "num_node_upper_bound": 20,
            "node_sample_probability": 0.15,
        },
    )

    env.reset()
    metrics, decision_event, done = env.step(None)
    while not done:
        assert isinstance(decision_event, MISDecisionPayload)
        vertex_state = decision_event.vertex_states
        undecided_mask = vertex_state == VertexState.Deferred
        random_mask = torch.rand(vertex_state.size(), device=device) < 0.8
        vertex_state[undecided_mask & random_mask] = VertexState.Included
        action = Action(vertex_state)
        metrics, decision_event, done = env.step(action)

        for key in [MISEnvMetrics.IncludedNodeCount]:
            print(f"[{env.tick - 1:02d}] {key:28s} {metrics[key][env.tick - 1].reshape(-1)}")
        for key in [MISEnvMetrics.IsDoneMasks]:
            print(f"[{env.tick - 1:02d}] {key:28s} {metrics[key].reshape(-1)}")
