import os


def get_experiment_data_stream(experiment_name: str):
    is_streamable_enabled: bool = bool(
        os.environ.get("MARO_STREAMABLE_ENABLED", False))

    stream = None

    if not is_streamable_enabled:

        def dummy(self, *args, **kwargs):
            pass

        class ExperimentDataStreamDummy:
            def __getattr__(self, name):
                return dummy

        stream = ExperimentDataStreamDummy()
    else:
        from .stream import ExperimentDataStream

        stream = ExperimentDataStream(experiment_name)

    return stream


__all__ = ["get_experiment_data_stream"]
