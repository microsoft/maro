from maro.rl_v3.distributed.dispatcher import Dispatcher
from maro.rl_v3.utils.common import from_env

Dispatcher(from_env("DISPATCHER_HOST"), from_env("NUM_WORKERS")).start()
