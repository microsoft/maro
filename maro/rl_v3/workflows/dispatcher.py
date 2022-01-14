from maro.rl_v3.distributed.dispatcher import Dispatcher
from maro.rl_v3.utils.common import from_env_as_int

dispatcher = Dispatcher(
    from_env_as_int("NUM_WORKERS"),
    frontend_port=from_env_as_int("DISPATCHER_FRONTEND_PORT"),
    backend_port=from_env_as_int("DISPATCHER_BACKEND_PORT")
)
dispatcher.start()
