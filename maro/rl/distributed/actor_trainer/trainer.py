# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from typing import Callable, List, Union

import numpy as np

from maro.communication import Proxy, RegisterTable, SessionMessage
from maro.rl.agent.abs_agent_manager import AbsAgentManager

from .common import Component, MessageTag, PayloadKey

TRAINER = Component.TRAINER.value


class Trainer(object):
    """Trainer is responsible for training models using experiences from actors.

    Args:
        agent_manager (AbsAgentManager): An AgentManager instance that manages all agents.
        experience_collection_func (Callable): Function to collect experiences from multiple remote actors.
        proxy_params: Parameters required for instantiating an internal proxy for communication.
    """

    def __init__(
        self,
        agent_manager: AbsAgentManager,
        experience_collecting_func: Callable,
        update_trigger: str = None,
        **proxy_params
    ):
        super().__init__()
        self._agent_manager = agent_manager
        self._experience_collecting_func = experience_collecting_func
        self._proxy = Proxy(component_type=TRAINER, **proxy_params)
        self._num_actors = len(self._proxy.peers_name["actor"])
        self._exploration_params_by_actor = defaultdict(lambda: None)
        self._registry_table = RegisterTable(self._proxy.peers_name)
        self._registry_table.register_event_handler(
            f"actor:{MessageTag.EXPLORATION_PARAMS.value}:1", self._update_exploration_params
        )
        if update_trigger is None:
            update_trigger = self._num_actors
        self._registry_table.register_event_handler(
            f"actor:{MessageTag.UPDATE.value}:{update_trigger}", self._update
        )

    def launch(self):
        for msg in self._proxy.receive():
            self._registry_table.push(msg)
            for handler_fn, cached_messages in self._registry_table.get():
                handler_fn(cached_messages)

    def _update_exploration_params(self, message):
        actor_id = message.session_id.split(".")[0]
        self._exploration_params_by_actor[actor_id] = message.payload[PayloadKey.EXPLORATION_PARAMS]
        self._proxy.reply(received_message=message, tag=MessageTag.EXPLORATION_PARAMS_ACK)

    def _update(self, messages):
        experiences_by_agent = {msg.source: msg.payload[PayloadKey.EXPERIENCES] for msg in messages}
        self._agent_manager.train(self._experience_collecting_func(experiences_by_agent))
        for msg in messages:
            self._proxy.reply(
                received_message=msg,
                tag=MessageTag.MODEL,
                payload={PayloadKey.MODEL: self._agent_manager.dump_models()}
            )

    def dump_models(self, dir_path: str):
        self._agent_manager.dump_models_to_files(dir_path)


class SEEDTrainer(Trainer):
    """Subclass of ``Trainer`` based on the SEED RL architecture.

    See https://arxiv.org/pdf/1910.06591.pdf for experiences.

    Args:
        agent_manager (AbsAgentManager): An AgentManager instance that manages all agents.
        experience_collection_func (Callable): Function to collect experiences from multiple remote actors.
        proxy_params: Parameters required for instantiating an internal proxy for communication.
    """
    def __init__(
        self,
        agent_manager: AbsAgentManager,
        experience_collecting_func: Callable,
        update_trigger: str = None,
        choose_action_trigger: str = None,
        **proxy_params
    ):
        super().__init__(agent_manager, experience_collecting_func, update_trigger=update_trigger, **proxy_params)
        if choose_action_trigger is None:
            choose_action_trigger = self._num_actors
        self._registry_table.register_event_handler(
            f"actor:{MessageTag.ACTION.value}:{choose_action_trigger}", self._get_action
        )

    def _get_action(self, messages: Union[List[SessionMessage], SessionMessage]):
        if isinstance(messages, SessionMessage):
            messages = [messages]
        # If there is no exploration parameters, use batch inference.
        if not self._exploration_params_by_actor:
            # group messages from different actors by the AGENT_ID field
            messages_by_agent_id = defaultdict(list)
            for msg in messages:
                messages_by_agent_id[msg.payload[PayloadKey.AGENT_ID]].append(msg)

            # batch inference for each agent_id
            for agent_id, message_batch in messages_by_agent_id.items():
                state_batch = np.vstack([msg.payload[PayloadKey.STATE] for msg in message_batch])
                action_batch = self._agent_manager[agent_id].choose_action(state_batch)
                for msg, action in zip(message_batch, action_batch):
                    self._proxy.reply(received_message=msg, tag=MessageTag.ACTION, payload={PayloadKey.ACTION: action})
        else:
            for msg in messages:
                actor_id = msg.session_id.split(".")[0]
                agent_id = msg.payload[PayloadKey.AGENT_ID]
                if self._exploration_params_by_actor[actor_id] is not None:
                    self._agent_manager.update_exploration_params(self._exploration_params_by_actor[actor_id])
                model_action = self._agent_manager[agent_id].choose_action(msg.payload[PayloadKey.STATE])
                self._proxy.reply(
                    received_message=msg, tag=MessageTag.ACTION, payload={PayloadKey.ACTION: model_action}
                )

    def _update(self, messages):
        experiences_by_agent = {msg.source: msg.payload[PayloadKey.EXPERIENCES] for msg in messages}
        self._agent_manager.train(self._experience_collecting_func(experiences_by_agent))
        for msg in messages:
            self._proxy.reply(received_message=msg, tag=MessageTag.TRAINING_FINISHED)
