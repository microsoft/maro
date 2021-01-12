
import psycopg2
import warnings

from datetime import datetime

from .common import MessageType, DataType
from multiprocessing import Process, Queue


class ExperimentDataSender(Process):
    """Experiment data sending process."""

    def __init__(self, data_queue: Queue, experiment_name, address, port, user, password, database):
        super().__init__()

        self._address = address
        self._port = port
        self._user = user
        self._password = password
        self._database = database
        self._cur_experiment = experiment_name

        self._data_queue = data_queue

        self._category_headers = {}
        self._values_pattern = {}

    def run(self):
        conn = None
        cursor = None

        try:
            print("Connecting to data base server....")

            conn = psycopg2.connect(
                user=self._user,
                password=self._password,
                host=self._address,
                port=self._port,
                database=self._database
            )

            cursor = conn.cursor()

            self._prepare_database(conn, cursor)

            print("Connected.")

            print("Pulling and Pushing...")

            while True:
                try:
                    data = self._data_queue.get(timeout=1)

                    if data == "stop":
                        return
                except Exception as ex:
                    warnings.warn(str(ex))

                # send data
                self._send_data(conn, cursor, data)
  
        except Exception as ex:
            warnings.warn(str(ex))
        finally:
            if cursor:
                cursor.close()

            if conn:
                conn.close()


        if self._data_queue.qsize() > 0:
            self._data_queue.close()

    def _prepare_database(self, conn, cursor):
        # make sure table same as current experiment name
        if conn and cursor:
            # fetch existing tables
            cursor.execute("all_tables();")

            is_experiments_table_exist = False
            is_current_experiment_exist = False

            # the result is something like: [(table_name,), (table_name,)]
            for table in cursor.fetchall():
                table_name = table[0]

                if table_name == "maro.experiments":
                    is_experiments_table_exist = True

            # make sure there is an "maro.experiments" table that use to hold experiment base info
            if not is_experiments_table_exist:
                cursor.execute("CREATE TABLE 'maro.experiments' (name STRING, scenario STRING, topology STRING, durations INT, episodes INT, created_at TIMESTAMP);")

    def _send_data(self, conn, cursor, msg):
        msg_type, data = msg

        if msg_type == MessageType.BeginExperiment:
            cursor.execute(
                """INSERT INTO "maro.experiments" (name, scenario, topology, durations, episodes, created_at) VALUES (%s, %s, %s, %s, %s, %s);""",
                (self._cur_experiment, data[0], data[1], data[4], data[2], int(datetime.timestamp(datetime.now()) * 1000000))
            )

            conn.commit()
        elif msg_type == MessageType.Data:
            episode, tick, data_list = data

            for data in data_list:
                category = data[0]

                values = list(data[1:])
                values.extend([tick, episode, (episode << 16) | tick])

                cursor.execute(
                    f"INSERT INTO \"{self._cur_experiment}.{category}\" ({self._category_headers[category]}) VALUES ({self._values_pattern[category]})",
                    tuple(values)
                )

            conn.commit()

        elif msg_type == MessageType.Category:
            category_name, fields = data
            table_name = f"{self._cur_experiment}.{category_name}"

            # TODO: data type checking

            # insert a _ts column as our timestamp, _tick and _episode column as our index
            # fields["_ts"] = DataType.Timestamp
            # fields["_tick"] = DataType.Int
            # fields["_episode"] = DataType.Int
            fields_declaration = [f"{fname} {ftype.value}" for fname, ftype in fields.items()]
            fields_declaration.extend(["_tick INT", "_episode INT", "_ts TIMESTAMP"])
            field_name_list = [f for f in fields.keys()] + ["_tick", "_episode", "_ts"]
            self._category_headers[category_name] = ",".join(field_name_list)
            self._values_pattern[category_name] = ",".join(["%s"] * len(field_name_list))
            field_declare = ",".join(fields_declaration)

            create_table_sql = f"CREATE TABLE '{table_name}' ({field_declare});"

            cursor.execute(create_table_sql)

            conn.commit()
