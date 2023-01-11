# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.event_buffer import EventBuffer, EventState
from maro.simulator.scenarios import AbsBusinessEngine

backends_to_test = ["static", "dynamic"]


def next_step(eb: EventBuffer, be: AbsBusinessEngine, tick: int):
    if tick > 0:
        # lets post process last tick first before start a new tick
        is_done = be.post_step(tick - 1)

        if is_done:
            return True

    be.step(tick)

    pending_events = eb.execute(tick)

    if len(pending_events) != 0:
        for evt in pending_events:
            evt.state = EventState.FINISHED

        eb.execute(tick)

    be.frame.take_snapshot(tick)

    return False


def be_run_to_end(eb, be):
    is_done = False

    tick = 0

    while not is_done:
        is_done = next_step(eb, be, tick)
        tick += 1


def compare_list(list1: list, list2: list) -> bool:
    return len(list1) == len(list2) and all(val1 == val2 for val1, val2 in zip(list1, list2))


def compare_dictionary(dict1: dict, dict2: dict) -> bool:
    keys1 = sorted(list(dict1.keys()))
    keys2 = sorted(list(dict2.keys()))
    if not compare_list(keys1, keys2):
        return False

    for key in keys1:
        value1 = dict1[key]
        value2 = dict2[key]
        if type(value1) != type(value2):
            return False
        if type(value1) == dict:
            if not compare_dictionary(value1, value2):
                return False
        elif type(value1) == list:
            if not compare_list(value1, value2):
                return False
        else:
            if value1 != value2:
                return False
    return True
