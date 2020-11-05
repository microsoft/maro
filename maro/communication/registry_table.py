# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# native lib
import itertools
from enum import Enum
from typing import List, Tuple, Union

import numpy as np

# private lib
from maro.utils.exception.communication_exception import ConditionalEventSyntaxError, PeersMissError

from .message import Message


class Operation(Enum):
    """The Enum class of the logic operations."""
    AND = "AND"
    OR = "OR"


class SuffixTree:
    """The suffix tree uses to decompose the conditional event.

    For the unit conditional event, the ``SuffixTree.value`` is the unit conditional event;
    For the conditional event, the ``SuffixTree.value`` is one of the ``Operation``, and the
    ``SuffixTree.nodes`` is the list of the unit conditional event.

    Args:
        value (Operation|str): Event operation: ``Operation.AND`` or ``Operation.OR``, or the unit conditional event.
        nodes List[SuffixTree]: List of the SuffixTrees.
    """

    def __init__(self, value=None, nodes=None):
        self.value = value
        self.nodes = nodes if nodes else []


class ConditionalEvent:
    """The description of the messages' combination.

    The conditional event can be composed of any number of unit conditional events and end with an Operation.
    For unit conditional event, It must be three parts and divided by ``:``:

    - The first part of unit event represent the message's source. E.g. ``learner`` or ``*``.
    - The second part of unit event represent the message's type. E.g. ``experience`` or ``*``.
    - The third part of unit event represent how much messages needed. E.g. ``1`` or ``90%``.

    Note:
        Do not use special symbol in the unit event, such as ``,``, ``(``, ``)``.

    Args:
        event (str|Tuple): The description of the requisite messages' combination.
            E.g. "actor:rollout:1" or ("learner:rollout:1", "learner:update:1", "AND").
        get_peers (callable): The callable function which returns the newest peer's name list from proxy.
    """

    def __init__(self, event: Union[str, Tuple], get_peers: callable):
        self._event = event
        self._get_peers = get_peers
        self._suffix_tree = SuffixTree()
        self._unit_event_message_dict = {}

        self._decomposer(self._event, self._suffix_tree)

    def _decomposer(self, event: Union[str, Tuple], suffix_tree: SuffixTree):
        """To generate suffix tree depending on the conditional event."""
        operation_and_list = ["&&", "AND"]
        operation_or_list = ["||", "OR"]

        # Check it is unit conditional event(str) or conditional event(tuple).
        if isinstance(event, str):
            self._unit_event_syntax_check(event)
            self._unit_event_message_dict[event] = []
            suffix_tree.value = event
        elif isinstance(event, tuple):
            if event[-1] in operation_and_list:
                suffix_tree.value = Operation.AND
            elif event[-1] in operation_or_list:
                suffix_tree.value = Operation.OR
            else:
                raise ConditionalEventSyntaxError(
                    "The last of the conditional event tuple must be one of ['AND', 'OR', '&&', '||]"
                )

            for slot in event[:-1]:
                node = SuffixTree()
                if isinstance(slot, tuple):
                    self._decomposer(slot, node)
                else:
                    self._unit_event_syntax_check(slot)
                    self._unit_event_message_dict[slot] = []
                    node.value = slot
                suffix_tree.nodes.append(node)
        else:
            raise ConditionalEventSyntaxError("Conditional event should be string or tuple.")

    def _unit_event_syntax_check(self, unit_event: str):
        """To check unit conditional event expression.

        Args:
            unit_event (str): The description of the requisite messages.
        """
        slots = unit_event.split(":")
        if len(slots) != 3:
            raise ConditionalEventSyntaxError(
                f"The conditional event: {unit_event}, must have three parts, and divided by ':'."
            )

        # The third part of unit conditional event must be an integer or percentage(*%).
        if slots[-1][-1] == "%":
            slots[-1] = slots[-1][:-1]

        try:
            int(slots[-1])
        except Exception as e:
            raise ConditionalEventSyntaxError(
                f"The third part of conditional event must be an integer or percentage with % in the end. {str(e)}"
            )

    def _get_request_message_number(self, unit_event: str) -> int:
        """To get the number of request messages by the given unit event."""
        component_type, _, number = unit_event.split(":")
        peers_number = len(self._get_peers(component_type))

        if peers_number == 0:
            raise PeersMissError(f"There is no target component in peer list! Required peer type {component_type}.")

        if number[-1] != "%":
            return int(number) if int(number) <= peers_number else peers_number
        else:
            request_message_number = np.floor(int(number[:-1]) * peers_number / 100)
            return int(request_message_number) if int(request_message_number) <= peers_number else peers_number

    def _unit_event_satisfied(self, unit_event: str) -> List[str]:
        """To check if the given unit conditional event has been satisfied.

        Returns:
            If the unit conditional event has been satisfied, it will return the list of unit conditional event;
            otherwise, it will return None.
        """
        request_message_number = self._get_request_message_number(unit_event)

        # Check if unit conditional event dict storied enough message.
        if request_message_number <= len(self._unit_event_message_dict[unit_event]):
            return [unit_event]

        return []

    def _conditional_event_satisfied(self, suffixtree) -> List[str]:
        """To check if the completed conditional event been satisfied."""
        operation = suffixtree.value
        if isinstance(operation, str):
            return self._unit_event_satisfied(operation)

        result = []
        for node in suffixtree.nodes:
            result.append(self._conditional_event_satisfied(node))

        for r in result:
            if operation == Operation.AND and not r:
                return []
            if operation == Operation.OR and r:
                return r

        # Flatten.
        flatten_result = list(itertools.chain.from_iterable(result))
        return flatten_result

    def push_message(self, message: Message):
        """Push message to all satisfied unit conditional events.

        Args:
            message (Message): Received message.
        """
        message_source, message_type = message.source, message.tag

        if isinstance(message_type, Enum):
            message_type = message_type.value

        for unit_event in self._unit_event_message_dict.keys():
            event_source, event_type, _ = unit_event.split(":")
            source_match, type_match = False, False
            if event_source == "*" or event_source in message_source:
                source_match = True
            if event_type == "*" or event_type == message_type:
                type_match = True

            if source_match and type_match:
                self._unit_event_message_dict[unit_event].append(message)

    def get_qualified_message(self) -> List[Message]:
        """Self-inspection of conditional event, if satisfied, pop qualified messages.

        Return:
            List[Message]: List of qualified messages.
        """
        message_list = []
        satisfied_unit_events = self._conditional_event_satisfied(self._suffix_tree)

        if satisfied_unit_events:
            for unit_event in satisfied_unit_events:
                request_message_number = self._get_request_message_number(unit_event)
                message_list.append(self._unit_event_message_dict[unit_event][:request_message_number])
                del self._unit_event_message_dict[unit_event][:request_message_number]

            # Flatten.
            message_list = list(itertools.chain.from_iterable(message_list))

        return message_list


class RegisterTable:
    """The RegisterTable is responsible for matching ``conditional events`` and user-defined ``message handlers``.

    Args:
        get_peers (callable): The callable function which returns the newest peer's name list from proxy.
    """

    def __init__(self, get_peers: callable):
        self._event_handler_dict = {}
        self._get_peers = get_peers

    def register_event_handler(self, event: Union[str, tuple], handler_fn: callable):
        """Register conditional event in the RegisterTable, and create a dict which match ``message handler`` and
        ``conditional event``.

        Args:
            event (str|Tuple): The description of the requisite messages' combination,
            handler_fn (callable): The user-define function which usually uses to handle incoming messages.
        """
        event = ConditionalEvent(event, self._get_peers)
        self._event_handler_dict[event] = handler_fn

    def push(self, message: Message):
        """Push message into all ``conditional events`` which register in the Registry Table.

        Args:
            message (Message): Received message.
        """
        for event in self._event_handler_dict.keys():
            event.push_message(message)

    def get(self) -> List[Tuple[callable, List[Message]]]:
        """If any ``conditional event`` has been satisfied, return the requisite message list and
        the correlational handler function.

        Return:
            List[Tuple[callable, List[Message]]]: The list of triggered handler functions and messages.
            E.g. [(handle_function_1, [messages]), (handle_function_2, [messages])]
        """
        satisfied_handler_fn = []

        for event, handler_fn in self._event_handler_dict.items():
            message_list = event.get_qualified_message()

            if message_list:
                if len(message_list) == 1:
                    satisfied_handler_fn.append((handler_fn, message_list[0]))
                else:
                    satisfied_handler_fn.append((handler_fn, message_list))

        return satisfied_handler_fn
