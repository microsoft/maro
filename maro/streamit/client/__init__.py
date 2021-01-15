import os


def streamit(experiment_name: str):
    is_streamable_enabled: bool = bool(
        os.environ.get("MARO_STREAMABLE_ENABLED", False))

    instance = None

    if not is_streamable_enabled:

        def dummy(self, *args, **kwargs):
            pass

        class DummyClient:
            def __getattr__(self, name):
                return dummy

        instance = DummyClient()
    else:
        from .client import Client

        instance = Client(experiment_name)

    return instance


__all__ = ["streamit"]
