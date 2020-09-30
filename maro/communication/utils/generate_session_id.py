# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid


def session_id_generator(source: str, destination: str) -> str:
    """
    The helper function to generate unique session id.

    Args:
        source (str): Message's source,
        destination (str): Message's destination.

    Returns:
        session_id (str): The unique session id.
            i.e. "uuid.source.destination"
    """
    unique_id = str(uuid.uuid4())
    session_id = '.'.join([unique_id, source, destination])

    return session_id
