# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.event_buffer import EventBuffer, EventState
from maro.simulator.scenarios import AbsBusinessEngine


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
        tick+=1
