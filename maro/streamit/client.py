from streamit.client import get_experiment_data_stream
import time
import os


os.environ["MARO_STREAMABLE_ENABLED"] = "True"


if __name__ == "__main__":

    start_time = time.perf_counter()

    stream = get_experiment_data_stream(f"test_expmt_{time.time()}")
    stream.start()

    total_eps = 1
    durations = 10
    port_num = 1
    vessel_num = 1

    stream.info("cim", "toy.1.1", durations,
                total_eps, config="<config></config>")

    for ep in range(total_eps):
        print(ep)

        stream.episode(ep)
        for tick in range(durations):
            stream.tick(tick)
            # ports
            for i in range(port_num):
                stream.data(
                    "port_detail", index=i, empty=i * 10, full=i * 100, capacity=i * 1000
                )

            for i in range(vessel_num):
                stream.data(
                    "vessel_detail", index=i, empty=i * 10, full=i * 100, capacity=i * 1000, remaining_space=i * 200
                )

    stream.close()

    print("total time cost:", time.perf_counter() - start_time)
