import os

client_instance = None


def streamit():
    is_streamable_enabled: bool = bool(
        os.environ.get("MARO_STREAMABLE_ENABLED", False)
    )

    global client_instance

    if client_instance is not None:
        return client_instance

    if not is_streamable_enabled:

        def dummy(self, *args, **kwargs):
            pass

        class DummyClient:
            def __getattr__(self, name):
                return dummy

        client_instance = DummyClient()
    else:
        from .client import Client

        client_instance = Client()

    return client_instance


__all__ = ["streamit"]
