import os


def get_experiment_data_stream(experiment_name: str):
    is_streamable_enabled: bool = bool(
        os.environ.get("MARO_STREAMABLE_ENABLED", False))

    streamit = None

    if not is_streamable_enabled:

        def dummy(self, *args, **kwargs):
            pass

        class DummyStreamit:
            def __getattr__(self, name):
                return dummy

        streamit = DummyStreamit()
    else:
        from .streamit import Streamit

        streamit = Streamit(experiment_name)

    return streamit


__all__ = ["get_experiment_data_stream"]
