
from datetime import datetime
import asyncio
import warnings
import psycopg2

from multiprocessing import Process, Queue
from .metric import Metric
from .common import MessageType


NEXT_LINE = bytes("\n", "utf-8")


# class BaseBatch:
#     def __init__(self, total_episodes: int, durations: int):
#         self._total_episodes = total_episodes
#         self._durations = durations

#         self._metric_collection = MetricCollection()

#     def add_data(self, metric: Metric):
#         """Add data to cache"""
#         self._metric_collection.append(metric)

#     def get_data_to_send(self, episode: int, tick: int, is_force=False):
#         """Return None is no data need to send"""
#         pass

# class DataNumberBatch(BaseBatch):
#     def __init__(self, total_episodes: int, durations: int, thredhold: int):
#         super().__init__(total_episodes, durations)

#         self._thredhold = thredhold

#     def get_data_to_send(self, episode: int, tick: int, is_force=False):
#         if len(self._metric_collection) >= self._thredhold or is_force or tick >= self._durations:
#             data_to_send = self._metric_collection

#             self._metric_collection = MetricCollection()

#             return data_to_send

#         return None

# class TicksBatch(BaseBatch):
#     pass

# class EpisodeBatch(BaseBatch):
#     pass

class StreamitSender(Process):
    def __init__(self, data_queue: Queue, experiment_name: str, address: str):
        super().__init__()

        self._address = address
        self._experiment_name = experiment_name
        self._data_queue = data_queue

        self._cur_episode = 0
        self._cur_tick = 0

        self._table_name_dict = {}

    def run(self):
        loop = asyncio.get_event_loop()

        loop.run_until_complete(self._start(loop))

        try:
            loop.run_forever()
        except Exception as e:
            print(e)

    async def _start(self, loop: asyncio.AbstractEventLoop):
        writer: asyncio.StreamWriter = None
        reader: asyncio.StreamReader = None
        conn = None
        cursor = None

        try:
            reader, writer = await asyncio.open_connection(host=self._address, port=9009, loop=loop)
            conn = psycopg2.connect(
                database="qdb", user="admin", password="quest", host=self._address, port="8812")
            cursor = conn.cursor()
        except Exception as ex:
            print(ex)

            loop.stop()

            return

        metrics = []
        is_stopping = False

        while True:
            try:
                msg = self._data_queue.get(timeout=1)

                msg_type, data = msg

                if msg_type == MessageType.Experiment:
                    expmt_metric = Metric("maro.experiments")
                    expmt_metric.with_timestamp(
                        datetime.timestamp(datetime.now()) * 1e9)

                    expmt_metric.add_value("name", self._experiment_name)
                    expmt_metric.add_value("scenario", data[0])
                    expmt_metric.add_value("topology", data[1])
                    expmt_metric.add_value("duratios", data[2])
                    expmt_metric.add_value("total_episodes", data[3])

                    # any additional data?
                    if len(data) > 4:
                        for k, v in data[4].items():
                            expmt_metric.add_value(k, v)

                    await self._send(writer, [expmt_metric])
                elif msg_type == MessageType.Episode:
                    # force to send data
                    self._cur_episode = data
                elif msg_type == MessageType.Tick:
                    # send data before new tick
                    if len(metrics) > 0:
                        await self._send(writer, metrics)

                        metrics = []

                    self._cur_tick = data
                elif msg_type == MessageType.Data:
                    category, data_dict = data

                    metric = Metric(f"{self._experiment_name}.{category}")

                    for k, v in data_dict.items():
                        metric.add_value(k, v)

                    metric.add_tag("Epsode", self._cur_episode)
                    metric.add_tag("Tick", self._cur_tick)
                    metric.add_tag("Experiment", self._experiment_name)
                    metric.with_timestamp(
                        (self._cur_episode << 32) | self._cur_tick)

                    metrics.append(metric)

                    if len(metrics) > 6000:
                        await self._send(writer, metrics)

                        metrics = []
                elif msg_type == MessageType.BigString:
                    pass
                    # NOTE: for big data, we only support same type
                    # category, data_dict = data

                    # table_name = f"{self._experiment_name}.{category}"

                    # if category not in self._table_name_dict:
                    #     field_name_list = [k for k in data_dict.keys()]

                    #     field_name_list.extend(["Episode", "Tick"])

                    #     field_def = ",".join(
                    #         [f"{k} STRING" for k in field_name_list])
                    #     cursor.execute(
                    #         f"CREATE TABLE '{table_name}' ({field_def})")

                    #     self._table_name_dict[category] = ",".join(
                    #         k for k in data_dict.keys())

                    # values = [v for v in data_dict.values()]
                    # values.extend([self._cur_episode, self._cur_tick])

                    # value_format = ",".join(["%s"] * len(values))
                    # print("sending big string", values)
                    # cursor.execute(f"INSERT INTO \"{table_name}\" ({self._table_name_dict[category]}) VALUES ({value_format})",
                    #                tuple(values))
                    # print("done")

                elif msg_type == MessageType.Close:
                    is_stopping = True
                else:
                    warnings.warn(f"Invalid message type: {msg_type}")
            except Exception as ex:
                print(str(ex))
                if is_stopping:
                    print("stopping..")
                    break

                continue

        if len(metrics) > 0:
            await self._send(writer, metrics)

        if writer:
            writer.close()

        if conn:
            cursor.close()
            conn.close()

        loop.stop()

    async def _send(self, writer: asyncio.StreamWriter, metrics: list):
        if metrics and len(metrics) > 0:
            msg_str = "\n".join([str(m) for m in metrics])

            writer.write(bytes(msg_str, "utf-8"))

            # NOTE: we should supply a \n to completed influxdb line protocol message
            writer.write(NEXT_LINE)

            await writer.drain()
