import os

from maro.rl_v3.distributed.train_ops_worker import TrainOpsWorker
from maro.rl_v3.workflows.test.ops_creator import ops_creator

TrainOpsWorker(os.getenv("ID"), ops_creator, "127.0.0.1").start()
