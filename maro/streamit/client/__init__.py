import os
import time

import traceback

from multiprocessing import Value

streamit = None

print("streamit?", os.environ["MARO_STREAMIT_ENABLED"])

is_streamable_enabled: bool = os.environ.get("MARO_STREAMIT_ENABLED", "") == "true"

experiment_name: str = os.environ.get("MARO_STREAMIT_EXPERIMENT_NAME", f"UNNAMED_EXPERIMENT_{time.time()}")

server_ip = os.environ.get("MARO_STREAMIT_SERVER_IP", "127.0.0.1")

if streamit is None:
    if not is_streamable_enabled:
        def dummy(self, *args, **kwargs):
            pass

        class DummyClient:
            def __getattr__(self, name):
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
        print("start sending service.")
        # traceback.print_stack()
        from .client import Client

        streamit = Client()

        streamit.start(experiment_name, server_ip)

__all__ = ["streamit"]
