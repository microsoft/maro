# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from collections import defaultdict
from typing import Callable, List, Union

from maro.communication import Message, Proxy, RegisterTable, SessionType
from maro.rl.agent import AbsAgent, MultiAgentWrapper
from maro.rl.scheduling.scheduler import Scheduler
from maro.utils import InternalLogger

from .message_enums import MessageTag, PayloadKey


class Learner(object):
    """Learner class.

    Args:
        group_name (str): Identifier of the group to which the actor belongs. It must be the same group name
            assigned to the actors (and roll-out clients, if any).
        num_actors (int): Expected number of actors in the group identified by ``group_name``.
        agent (Union[AbsAgent, MultiAgentWrapper]): Agent or ditionary of agents managed by the agent.
        scheduler (AbsScheduler): A scheduler responsible for iterating over episodes and generating exploration
            parameters if necessary.
        update_trigger (str): Number or percentage of ``MessageTag.FINISHED`` messages required to trigger
            learner updates, i.e., model training.
        inference (bool): If true, inference (i.e., action decisions) will be performed on the learner side.
            See https://arxiv.org/pdf/1910.06591.pdf for details. Defaults to False.
        inference_trigger (str): Number or percentage of ``MessageTag.CHOOSE_ACTION`` messages required to tigger
            batch inference.
        state_batching_func (Callable): A function to batch state objects from multiple roll-out clients
            for batch inference. Ignored if ``inference`` is false.
        proxy_options (dict): Keyword parameters for the internal ``Proxy`` instance. See ``Proxy`` class
            for details. Defaults to None.
    """
    def __init__(
        self,
        group_name: str,
        num_actors: int,
        agent: Union[AbsAgent, MultiAgentWrapper],
        scheduler: Scheduler,
        experience_pool,
        *,
        post_collect_callback: Callable,
        update_trigger: str = None,
        inference: bool = False,
        inference_trigger: str = None,
        state_batching_func: Callable = None,
        proxy_options: dict = None
    ):
        super().__init__()
        self.agent = agent
        self.scheduler = scheduler
        self.experience_pool = experience_pool
        self.post_collect_callback = post_collect_callback
        self.inference = inference
        peers = {"actor": num_actors}
        if inference:
            peers["decision_client"] = num_actors
        if proxy_options is None:
            proxy_options = {}
        self._proxy = Proxy(group_name, "learner", peers, **proxy_options)
        self._actors = self._proxy.peers_name["actor"]  # remote actor ID's
        self._registry_table = RegisterTable(self._proxy.peers_name)
        if update_trigger is None:
            update_trigger = len(self._actors)
        self._registry_table.register_event_handler(
            f"actor:{MessageTag.FINISHED.value}:{update_trigger}", self._on_rollout_finish
        )
        if inference:
            self._decision_clients = self._proxy.peers_name["decision_client"]
            if inference_trigger is None:
                inference_trigger = len(self._decision_clients)
            self._registry_table.register_event_handler(
                f"decision_client:{MessageTag.CHOOSE_ACTION.value}:{inference_trigger}", self._on_action_request
            )
            self._state_batching_func = state_batching_func
        else:
            self._decision_clients = None
            self._state_batching_func = None

        self._logger = InternalLogger(self._proxy.name)

    def run(self):
        for exploration_params in self.scheduler:
            metrics_by_src, exp_by_src = self.collect(
                self.scheduler.iter,
                model_dict=None if self.inference else self.agent.dump_model(),
                exploration_params=None if self.inference else exploration_params
            )
            self.on_collect(metrics_by_src, exp_by_src, self.experience_pool)
            self.agent.learn()

            self._logger.info("Training finished")

    def collect(self, rollout_index: int, training: bool = True, **rollout_kwargs) -> tuple:
        """Collect roll-out performances and details from remote actors.

        Args:
            rollout_index (int): Index of roll-out requests.
            training (bool): If true, the roll-out request is for training purposes.
            rollout_kwargs: Keyword parameters required for roll-out. Must match the keyword parameters specified
                for the roll-out executor.
        """
        payload = {
            PayloadKey.ROLLOUT_INDEX: rollout_index,
            PayloadKey.TRAINING: training,
            PayloadKey.ROLLOUT_KWARGS: rollout_kwargs
        }
        # If no actor client is found, it is necessary to broadcast agent models to the remote actors
        # so that thay can perform inference on their own. If there exists exploration parameters, they
        # must also be sent to the remote actors.
        self._proxy.iscatter(MessageTag.ROLLOUT, SessionType.TASK, [(actor, payload) for actor in self._actors])
        self._logger.info(f"Sent roll-out requests to {self._actors} for ep-{rollout_index}")

        # Receive roll-out results from remote actors
        for msg in self._proxy.receive():
            if msg.payload[PayloadKey.ROLLOUT_INDEX] != rollout_index:
                self._logger.info(
                    f"Ignore a message of type {msg.tag} with ep {msg.payload[PayloadKey.ROLLOUT_INDEX]} "
                    f"(current ep: {rollout_index})"
                )
                continue
            if msg.tag == MessageTag.FINISHED:
                # If enough update messages have been received, call update() and break out of the loop to start
                # the next episode.
                result = self._registry_table.push(msg)
                if result:
                    env_metrics, details = result[0]
                    break
            elif msg.tag == MessageTag.CHOOSE_ACTION:
                self._registry_table.push(msg)

        return env_metrics, details

    def _on_rollout_finish(self, messages: List[Message]):
        metrics = {msg.source: msg.payload[PayloadKey.METRICS] for msg in messages}
        details = {msg.source: msg.payload[PayloadKey.DETAILS] for msg in messages}
        return metrics, details

    def _on_action_request(self, messages: List[Message]):
        # group messages from different actors by the AGENT_ID field
        states_by_agent, msg_indexes_by_agent = defaultdict(list), defaultdict(list)
        for i, msg in enumerate(messages):
            for agent_id, state in msg.payload[PayloadKey.STATE].items():
                states_by_agent[agent_id].append(state)
                msg_indexes_by_agent[agent_id].append(i)

        # batch inference for each agent_id
        states_by_agent = {
            agent_id: self._state_batching_func(states) for agent_id, states in states_by_agent.items()
        }
        action_info_by_agent = self.agent.choose_action(states_by_agent)
        replies = [{} for _ in range(len(messages))]
        for agent_id, action_info in action_info_by_agent.items():
            if isinstance(action_info, tuple):
                action_info = list(zip(*action_info))
            assert len(msg_indexes_by_agent[agent_id]) == len(action_info)
            for idx, act in zip(msg_indexes_by_agent[agent_id], action_info):
                replies[idx][agent_id] = act
        
        for msg, rep in zip(messages, replies):
            self._proxy.reply(
                msg,
                tag=MessageTag.ACTION,
                payload={
                    PayloadKey.ACTION: rep,
                    PayloadKey.ROLLOUT_INDEX: msg.payload[PayloadKey.ROLLOUT_INDEX],
                    PayloadKey.TIME_STEP: msg.payload[PayloadKey.TIME_STEP]
                }
            )

    def exit(self):
        """Tell the remote actor to exit."""
        self._proxy.ibroadcast(
            component_type="actor", tag=MessageTag.EXIT, session_type=SessionType.NOTIFICATION
        )
        self._logger.info("Exiting...")
        sys.exit(0)
