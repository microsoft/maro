from streamit.client import get_experiment_data_stream
from streamit.client.common import DataType
import time
import os


os.environ["MARO_STREAMABLE_ENABLED"] = "True"


if __name__ == "__main__":

    start_time = time.time()

    stream = get_experiment_data_stream(f"test_expmt_{time.time()}")
    stream.start()

    total_eps = 10
    durations = 102

    stream.experiment_info("cim", "toy.1.1", total_eps, durations)
    stream.category(
        "port_detail",
        {
            "index": DataType.Int,
            "empty": DataType.Int,
            "full": DataType.Int,
            "shortage": DataType.Int
        })

    for ep in range(total_eps):
        print(ep)

        stream.episode(ep)
        for tick in range(durations):
            stream.tick(tick)
                # ports
            for i in range(10):
                stream.row(
                    "port_detail", i, i * 10, i * 100, i * 1000
                )
            # time.sleep(1)

    stream.close()

    print("total time cost:", time.time() - start_time)
