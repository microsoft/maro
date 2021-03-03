# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import os
import warnings
from datetime import datetime
from multiprocessing import Process, Queue

from .common import MessageType
from .metric import Metric

# We disable streamit here as we are in another process now,
# or it will try to start the sender again.
os.environ["MARO_STREAMIT_ENABLED"] = "false"


MAX_DATA_CACHE_NUMBER = 5000
NEXT_LINE = bytes("\n", "utf-8")


class StreamitSender(Process):
    """Process that used to send data to data services.

    Args:
        data_queue (Queue): Queue used to pass data from environment process to current.
        experiment_name (str): Name of current experiment.
        address (str): IP address of data service.
    """
    def __init__(self, data_queue: Queue, experiment_name: str, address: str):
        super().__init__()

        self._address = address
        self._experiment_name = experiment_name
        self._data_queue = data_queue

        self._cur_episode = 0
        self._cur_tick = 0

    def run(self):
        """Entry point of this process."""
        loop = asyncio.get_event_loop()

        loop.run_until_complete(self._start(loop))

        try:
            loop.run_forever()
        except Exception as e:
            print(e)

    async def _start(self, loop: asyncio.AbstractEventLoop):
        writer: asyncio.StreamWriter
        reader: asyncio.StreamReader

        try:
            reader, writer = await asyncio.open_connection(host=self._address, port=9009, loop=loop)
        except Exception as ex:
            print(ex)

            loop.stop()

            return

        # Message cache.
        metrics = []

        # If we received stop message?
        is_stopping = False

        while True:
            try:
                msg = self._data_queue.get(timeout=1)

                msg_type, data = msg

                if msg_type == MessageType.Experiment:
                    expmt_metric = Metric("maro.experiments")

                    expmt_metric.with_timestamp(int(datetime.timestamp(datetime.now()) * 1e9))
                    expmt_metric.add_value("name", self._experiment_name)
                    expmt_metric.add_value("scenario", data[0])
                    expmt_metric.add_value("topology", data[1])
                    expmt_metric.add_value("durations", data[2])

                    # Any additional data?
                    if len(data) > 4:
                        for k, v in data[4].items():
                            expmt_metric.add_value(k, v)

                    await self._send(writer, [expmt_metric])
                elif msg_type == MessageType.Episode:
                    self._cur_episode = data
                elif msg_type == MessageType.Tick:
                    # Send data before new tick.
                    if len(metrics) > 0:
                        await self._send(writer, metrics)

                        metrics = []

                    self._cur_tick = data
                elif msg_type == MessageType.Data:
                    category, data_dict = data

                    metric = Metric(f"{self._experiment_name}.{category}")

                    for k, v in data_dict.items():
                        metric.add_value(k, v)

                    metric.add_tag("episode", self._cur_episode)
                    metric.add_tag("tick", self._cur_tick)
                    metric.add_tag("experiment", self._experiment_name)
                    metric.with_timestamp((self._cur_episode << 32) | self._cur_tick)

                    metrics.append(metric)

                    if len(metrics) > MAX_DATA_CACHE_NUMBER:
                        await self._send(writer, metrics)

                        metrics = []
                elif msg_type == MessageType.Close:
                    is_stopping = True
                else:
                    warnings.warn(f"Invalid message type: {msg_type}")
            except Exception:
                if is_stopping:
                    break

                continue

        # Clear cache if there is any data.
        if len(metrics) > 0:
            await self._send(writer, metrics)

        if writer:
            writer.close()

        loop.stop()

    async def _send(self, writer: asyncio.StreamWriter, metrics: list):
        if metrics and len(metrics) > 0:
            msg_str = "\n".join([str(m) for m in metrics])

            writer.write(bytes(msg_str, "utf-8"))

            # NOTE: we should supply a \n to completed influxdb line protocol message.
            writer.write(NEXT_LINE)

            await writer.drain()
