# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# native lib
import uuid
from enum import Enum
from typing import Union

# private lib
from maro.utils.exception.communication_exception import MessageSessionTypeError

from .utils import session_id_generator


class SessionType(Enum):
    """Communication session categories.

    - ``TASK``: Task session is used to trigger remote job(s).
    - ``NOTIFICATION``: Notification session is used to sync information to peers.
    """
    TASK = "task"
    NOTIFICATION = "notification"


class TaskSessionStage(Enum):
    """Task session stages.

    - ``REQUEST``: Task session stage 1.
    - ``RECEIVE``: Task session stage 2.
    - ``COMPLETE``: Task session stage 3.
    """
    REQUEST = "task_request"
    RECEIVE = "task_receive"
    COMPLETE = "task_complete"


class NotificationSessionStage(Enum):
    """Notification session stages.

    - ``REQUEST``: Notification session stage 1.
    - ``RECEIVE``: Notification session stage 2.
    """
    REQUEST = "notification_request"
    RECEIVE = "notification_receive"


class Message(object):
    """General Message for hosting payload between receiver and sender.

    Args:
        tag (str|Enum): Message tag, which is customized by the user, for specific application logic.
        source (str): The sender of message.
        destination (str): The receiver of message.
        payload (object): Message payload, such as model parameters, experiences, etc. Defaults to None.
        session_id (str): Message belonged session id, it will be generated automatically by default, you can use it
            to group message based on your application logic.
    """

    def __init__(self, tag: Union[str, Enum], source: str, destination: str, payload=None, session_id: str = None):
        self.tag = tag
        self.source = source
        self.destination = destination
        self.payload = {} if payload is None else payload
        self.session_id = session_id if session_id else session_id_generator(self.source, self.destination)
        self.message_id = str(uuid.uuid1())

    def __repr__(self):
        return "; \n".join([f"{k} = {v}" for k, v in vars(self).items()])


class SessionMessage(Message):
    """The session message class.

    It is used by a specific session, which will contain session stage to support more complex application logic.

    Args:
        session_type (Enum): It indicates the current session type.
        session_stage (Enum): It indicates the current session stage.
    """

    def __init__(
        self, tag: Union[str, Enum], source: str, destination: str, payload=None,
        session_id: str = None, session_type: SessionType = SessionType.TASK, session_stage=None
    ):
        super().__init__(tag, source, destination, payload, session_id)
        self.session_type = session_type

        if self.session_type == SessionType.TASK:
            self.session_stage = session_stage if session_stage else TaskSessionStage.REQUEST
        elif self.session_type == SessionType.NOTIFICATION:
            self.session_stage = session_stage if session_stage else NotificationSessionStage.REQUEST
        else:
            raise MessageSessionTypeError(
                f"Receive unrecognized session type {self.session_type}, please use the SessionType class."
            )
