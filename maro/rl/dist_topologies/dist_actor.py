# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys

from maro.communication import Proxy, RegisterTable, SessionMessage
from maro.rl.storage.column_based_store import ColumnBasedStore

from .common import MessageTag, PayloadKey


class DistActor(object):
    def __init__(self, env, state_shaper, action_shaper, experience_shaper, proxy_params):
        self._env = env
        self._state_shaper = state_shaper
        self._action_shaper = action_shaper
        self._experience_shaper = experience_shaper
        self._proxy = Proxy(component_type="actor", **proxy_params)
        self._registry_table = RegisterTable(self._proxy.peers_name)
        self._registry_table.register_event_handler("learner:rollout:1", self._on_rollout_request)
        # self._registry_table.register_event_handler("learner:action:1", self._on_action_from_learner)
        # Data structures to temporarily store transitions and trajectory
        self._transition_cache = {}
        self._trajectory = ColumnBasedStore()

    def roll_out(self, return_details: bool = True):
        """Perform local roll-out and send the results back to the request sender.

        Args:
            done (bool): If True, the current call is the last call, i.e., no more roll-outs will be performed.
                This flag is used to signal remote actor workers to exit.
            return_details (bool): If True, return experiences as well as performance metrics provided by the env.
        """
        metrics, decision_event, is_done = self._env.step(None)
        while not is_done:
            model_action = self._choose_action(decision_event, self._env.snapshot_list)
            action = self._action_shaper(model_action)
            metrics, decision_event, is_done = self._env.step(action)
            self._transition_cache["metrics"] = metrics
            self._trajectory.put(self._transition_cache)

        details = self._post_process() if return_details else None
        return self._env.metrics, details

    def launch(self):
        """Entry point method.

        This enters the actor into an infinite loop of listening to requests and handling them according to the
        register table. In this case, the only type of requests the actor needs to handle is roll-out requests.
        """
        for msg in self._proxy.receive():
            self._registry_table.push(msg)
            triggered_events = self._registry_table.get()
            for handler_fn, cached_messages in triggered_events:
                handler_fn(cached_messages)

    def _on_rollout_request(self, message):
        data = message.payload
        if data.get(PayloadKey.DONE, False):
            sys.exit(0)

        performance, details = self.roll_out(return_details=message.payload[PayloadKey.RETURN_DETAILS])
        self._proxy.reply(
            received_message=message,
            tag=MessageTag.UPDATE,
            payload={PayloadKey.PERFORMANCE: performance, PayloadKey.DETAILS: details}
        )

    def _choose_action(self, decision_event, snapshot_list):
        agent_id, model_state = self._state_shaper(decision_event, snapshot_list)
        reply = self._proxy.send(
            SessionMessage(
                tag=MessageTag.CHOOSE_ACTION,
                source=self._proxy.component_name,
                destination=self._proxy.peers_name["learner"][0],
                payload={PayloadKey.STATE: model_state, PayloadKey.AGENT_ID: agent_id},
            )
        )
        model_action = reply[0].payload[PayloadKey.ACTION]
        self._transition_cache = {
            "state": model_state,
            "action": model_action,
            "reward": None,
            "agent_id": agent_id,
            "event": decision_event
        }

        return model_action

    def _post_process(self):
        """Process the latest trajectory into experiences."""
        experiences = self._experience_shaper(self._trajectory, self._env.snapshot_list)
        self._trajectory.clear()
        self._transition_cache = {}
        self._state_shaper.reset()
        self._action_shaper.reset()
        self._experience_shaper.reset()
        return experiences
