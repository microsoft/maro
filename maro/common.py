# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


class BaseDecisionEvent:
    """Base class for all decision events.

    We made this design for the convenience of users. As a summary, there are two types of events in MARO:
    - CascadeEvent & AtomEvent: used to drive the MARO Env / business engine.
    - DecisionEvent: exposed to users as a means of communication.

    The latter one serves as the `payload` of the former ones inside of MARO Env.

    Therefore, the related namings might be a little bit tricky.
    - Inside MARO Env: `decision_event` is actually a CascadeEvent. DecisionEvent is the payload of them.
    - Outside MARO Env (for users): `decision_event` is a DecisionEvent.
    """


class BaseAction:
    """Base class for all action payloads"""
