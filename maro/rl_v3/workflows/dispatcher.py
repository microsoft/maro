from maro.rl_v3.distributed.dispatcher import Dispatcher
from maro.rl_v3.utils.common import from_env

dispatcher = Dispatcher(
    from_env("NUM_WORKERS"),
    frontend_port=from_env("DISPATCHER_FRONTEND_PORT"),
    backend_port=from_env("DISPATCHER_BACKEND_PORT")
)
dispatcher.start()
