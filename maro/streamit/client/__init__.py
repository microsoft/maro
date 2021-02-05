# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import time

streamit = None

# We use environment variable to control if streaming enabled.
is_streamit_enabled: bool = os.environ.get("MARO_STREAMIT_ENABLED", "") == "true"

experiment_name: str = os.environ.get("MARO_STREAMIT_EXPERIMENT_NAME", "UNNAMED_EXPERIMENT")

# Append timestamp to all experiment name to make sure all experiment name are unique.
experiment_name = f"{experiment_name}.{time.time()}"

# Configure service host, but not port, as we hard coded the port for now.
server_ip = os.environ.get("MARO_STREAMIT_SERVER_IP", "127.0.0.1")

if streamit is None:
    # If not enabled, we return a dummy object that can accept any function/attribute call.
    if not is_streamit_enabled:
        # Function that use for dummy calling.
        def dummy(self, *args, **kwargs):
            pass

        class DummyClient:
            """Dummy client that hold call function call when disable streamit,
            to user do not need if-else for switching."""
            def __getattr__(self, name: str):
                return dummy

            def __bool__(self):
                return False

            def __enter__(self):
                """Support with statement."""
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                """Stop after exit with statement."""
                pass

        streamit = DummyClient()
    else:
        from .client import StreamitClient

        streamit = StreamitClient(experiment_name, server_ip)

__all__ = ["streamit"]
