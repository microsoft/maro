from maro.rl_v3.training.dispatcher import Dispatcher
from maro.rl_v3.utils.common import from_env_as_int

if __name__ == "__main__":
    dispatcher = Dispatcher(
        frontend_port=from_env_as_int("DISPATCHER_FRONTEND_PORT"),
        backend_port=from_env_as_int("DISPATCHER_BACKEND_PORT")
    )
    dispatcher.start()
