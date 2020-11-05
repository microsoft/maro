# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv
import os
import random
import zipfile
from enum import Enum

import geopy.distance
import numpy as np
import pandas as pd
from yaml import safe_load

from maro.cli.data_pipeline.base import DataPipeline, DataTopology
from maro.cli.data_pipeline.utils import StaticParameter, download_file
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


class CitiBikePipeline(DataPipeline):
    """Generate citi_bike data bin and other necessary files for the specified topology from specified source.

    They will be generated in ~/.maro/data/citi_bike/[topology]/_build.
    Folder structure:
    ~/.maro
            /data/citi_bike/[topology]
                                    /_build bin data file and other necessory files
                                    /source
                                            /_download original data files
                                            /_clean cleaned data files
            /temp download temp files

    Args:
        topology(str): Topology name of the data files.
        source(str): Source url of original data file.
        station_info(str): Source url of station info file.
        is_temp(bool): (optional) If the data file is temporary.
    """

    _download_file_name = "trips.zip"
    _station_info_file_name = "full_station.json"

    _clean_file_name = "trips.csv"

    _build_file_name = "trips.bin"
    _station_meta_file_name = "station_meta.csv"
    _distance_file_name = "distance_adj.csv"

    _meta_file_name = "trips.yml"

    def __init__(self, topology: str, source: str, station_info: str, is_temp: bool = False):
        super().__init__("citi_bike", topology, source, is_temp)

        self._station_info = station_info

        self._station_info_file = os.path.join(self._download_folder, self._station_info_file_name)

        self._distance_file = os.path.join(self._build_folder, self._distance_file_name)
        self._station_meta_file = os.path.join(self._build_folder, self._station_meta_file_name)

        self._common_data = {}

    def download(self, is_force: bool = False):
        """Download the zip file."""
        super().download(is_force)
        self._new_file_list.append(self._station_info_file)

        if (not is_force) and os.path.exists(self._station_info_file):
            logger.info_green("File already exists, skipping download.")
        else:
            logger.info_green(f"Downloading trip data from {self._station_info} to {self._station_info_file}.")
            download_file(source=self._station_info, destination=self._station_info_file)

    def clean(self):
        """Unzip the csv file and process it for building binary file."""
        super().clean()
        logger.info_green("Cleaning trip data.")
        if os.path.exists(self._download_file):
            # unzip
            logger.info_green("Unzip start.")
            with zipfile.ZipFile(self._download_file, "r") as zip_ref:
                for filename in zip_ref.namelist():
                    # Only one csv file is expected.
                    if (
                        filename.endswith(".csv") and
                        (not (filename.startswith("__MACOSX") or filename.startswith(".")))
                    ):

                        logger.info_green(f"Unzip {filename} from {self._download_file}.")
                        zip_ref.extractall(self._clean_folder, [filename])
                        unzip_file = os.path.join(self._clean_folder, filename)

                        self._new_file_list.append(unzip_file)
                        self._preprocess(unzipped_file=unzip_file)
                        break
        else:
            logger.warning(f"Not found downloaded trip data: {self._download_file}.")

    def _read_common_data(self):
        """Read and full init data and existed stations."""

        full_stations = None

        with open(self._station_info_file, mode="r", encoding="utf-8") as station_file:
            # read station to station file
            raw_station_data = pd.DataFrame.from_dict(pd.read_json(station_file)["data"]["stations"])
            station_data = raw_station_data.rename(columns={
                "lon": "station_longitude",
                "lat": "station_latitude",
                "region_id": "region"})

            # group by station to generate station init info
            full_stations = station_data[
                ["station_id", "capacity", "station_longitude", "station_latitude"]
            ].reset_index(drop=True)

            # generate station id by index

            full_stations["station_id"] = pd.to_numeric(full_stations["station_id"], downcast="integer")
            full_stations["capacity"] = pd.to_numeric(full_stations["capacity"], downcast="integer")
            full_stations["station_longitude"] = pd.to_numeric(full_stations["station_longitude"], downcast="float")
            full_stations["station_latitude"] = pd.to_numeric(full_stations["station_latitude"], downcast="float")
            full_stations.drop(full_stations[full_stations["capacity"] == 0].index, axis=0, inplace=True)
            full_stations.dropna(
                subset=["station_id", "capacity", "station_longitude", "station_latitude"], inplace=True
            )

        self._common_data["full_stations"] = full_stations

        self._common_data["full_station_num"] = len(self._common_data["full_stations"])
        self._common_data["full_dock_num"] = self._common_data["full_stations"]["capacity"].sum()

    def _read_src_file(self, file: str):
        """Read and return processed rows."""
        ret = []

        if os.path.exists(file):
            # For ignoring the unimportant issues in the source file.
            with open(file, "r", encoding="utf-8", errors="ignore") as fp:

                ret = pd.read_csv(fp)
                ret = ret[[
                    "tripduration", "starttime", "start station id", "end station id", "start station latitude",
                    "start station longitude", "end station latitude", "end station longitude", "gender", "usertype",
                    "bikeid"
                ]]
                ret["tripduration"] = pd.to_numeric(
                    pd.to_numeric(ret["tripduration"], downcast="integer") / 60, downcast="integer"
                )
                ret["starttime"] = pd.to_datetime(ret["starttime"])
                ret["start station id"] = pd.to_numeric(ret["start station id"], errors="coerce", downcast="integer")
                ret["end station id"] = pd.to_numeric(ret["end station id"], errors="coerce", downcast="integer")
                ret["start station latitude"] = pd.to_numeric(ret["start station latitude"], downcast="float")
                ret["start station longitude"] = pd.to_numeric(ret["start station longitude"], downcast="float")
                ret["end station latitude"] = pd.to_numeric(ret["end station latitude"], downcast="float")
                ret["end station longitude"] = pd.to_numeric(ret["end station longitude"], downcast="float")
                ret["bikeid"] = pd.to_numeric(ret["bikeid"], errors="coerce", downcast="integer")
                ret["gender"] = pd.to_numeric(ret["gender"], errors="coerce", downcast="integer")
                ret["usertype"] = ret["usertype"].apply(str).apply(
                    lambda x: 0 if x in ["Subscriber", "subscriber"] else 1 if x in ["Customer", "customer"] else 2
                )
                ret.dropna(subset=[
                    "start station id", "end station id", "start station latitude", "end station latitude",
                    "start station longitude", "end station longitude"
                ], inplace=True)
                ret.drop(
                    ret[
                        (ret["tripduration"] <= 1) |
                        (ret["start station latitude"] == 0) |
                        (ret["start station longitude"] == 0) |
                        (ret["end station latitude"] == 0) |
                        (ret["end station longitude"] == 0)
                    ].index,
                    axis=0,
                    inplace=True
                )
                ret = ret.sort_values(by="starttime", ascending=True)

        return ret

    def _process_src_file(self, src_data: pd.DataFrame):
        used_bikes = len(src_data[["bikeid"]].drop_duplicates(subset=["bikeid"]))

        trip_data = src_data[
            (src_data["start station latitude"] > 40.689960) &
            (src_data["start station latitude"] < 40.768334) &
            (src_data["start station longitude"] > -74.019623) &
            (src_data["start station longitude"] < -73.909760)
        ]
        trip_data = trip_data[
            (trip_data["end station latitude"] > 40.689960) &
            (trip_data["end station latitude"] < 40.768334) &
            (trip_data["end station longitude"] > -74.019623) &
            (trip_data["end station longitude"] < -73.909760)
        ]

        trip_data["start_station_id"] = trip_data["start station id"]
        trip_data["end_station_id"] = trip_data["end station id"]

        # get new stations
        used_stations = []
        used_stations.append(
            trip_data[["start_station_id", "start station latitude", "start station longitude", ]].drop_duplicates(
                subset=["start_station_id"]).rename(
                    columns={
                        "start_station_id": "station_id",
                        "start station latitude": "latitude",
                        "start station longitude": "longitude"
                    }))
        used_stations.append(
            trip_data[["end_station_id", "end station latitude", "end station longitude", ]].drop_duplicates(
                subset=["end_station_id"]).rename(
                    columns={
                        "end_station_id": "station_id",
                        "end station latitude": "latitude",
                        "end station longitude": "longitude"
                    }))

        in_data_station = pd.concat(used_stations, ignore_index=True).drop_duplicates(
            subset=["station_id"]
        ).sort_values(by=["station_id"]).reset_index(drop=True)

        stations_existed = pd.DataFrame(in_data_station[["station_id"]])

        stations_existed["station_index"] = pd.to_numeric(stations_existed.index, downcast="integer")

        # get start station id and end station id
        trip_data = trip_data.join(
            stations_existed.set_index("station_id"),
            on="start_station_id"
        ).rename(columns={"station_index": "start_station_index"})
        trip_data = trip_data.join(
            stations_existed.set_index("station_id"),
            on="end_station_id"
        ).rename(columns={"station_index": "end_station_index"})
        trip_data = trip_data.rename(columns={"starttime": "start_time", "tripduration": "duration"})

        trip_data = trip_data[
            ["start_time", "start_station_id", "end_station_id", "duration", "start_station_index", "end_station_index"]
        ]

        return trip_data, used_bikes, in_data_station, stations_existed

    def _process_current_topo_station_info(
            self, stations_existed: pd.DataFrame, used_bikes: int, loc_ref: pd.DataFrame):
        data_station_init = stations_existed.join(
            self._common_data["full_stations"][["station_id", "capacity"]].set_index("station_id"),
            on="station_id"
        ).join(
            loc_ref[["station_id", "latitude", "longitude"]].set_index("station_id"),
            on="station_id"
        )
        # data_station_init.rename(columns={"station_id": "station_index"}, inplace=True)
        avg_capacity = int(self._common_data["full_dock_num"] / self._common_data["full_station_num"])
        avalible_bike_rate = used_bikes / self._common_data["full_dock_num"]
        values = {"capacity": avg_capacity}
        data_station_init.fillna(value=values, inplace=True)
        data_station_init["init"] = (data_station_init["capacity"] * avalible_bike_rate).round().apply(int)
        data_station_init["capacity"] = pd.to_numeric(
            data_station_init["capacity"], errors="coerce", downcast="integer"
        )
        data_station_init["station_id"] = pd.to_numeric(
            data_station_init["station_id"], errors="coerce", downcast="integer"
        )

        return data_station_init

    def _process_distance(self, station_info: pd.DataFrame):
        distance_adj = pd.DataFrame(0, index=station_info["station_index"],
                                    columns=station_info["station_index"], dtype=np.float)
        look_up_df = station_info[["latitude", "longitude"]]
        return distance_adj.apply(lambda x: pd.DataFrame(x).apply(lambda y: geopy.distance.distance(
            (look_up_df.at[x.name, "latitude"], look_up_df.at[x.name, "longitude"]),
            (look_up_df.at[y.name, "latitude"], look_up_df.at[y.name, "longitude"])
        ).km, axis=1), axis=1)

    def _preprocess(self, unzipped_file: str):
        self._read_common_data()
        logger.info_green("Reading raw trip data.")
        org_data = self._read_src_file(file=unzipped_file)
        logger.info_green("Processing trip data.")
        trip_data, used_bikes, in_data_station, stations_existed = self._process_src_file(src_data=org_data)

        self._new_file_list.append(self._clean_file)
        self._new_file_list.append(self._station_meta_file)
        self._new_file_list.append(self._distance_file)

        with open(self._clean_file, mode="w", encoding="utf-8", newline="") as f:
            trip_data.to_csv(f, index=False, header=True)

        logger.info_green("Processing station info data.")
        station_info = self._process_current_topo_station_info(
            stations_existed=stations_existed, used_bikes=used_bikes, loc_ref=in_data_station
        )
        with open(self._station_meta_file, mode="w", encoding="utf-8", newline="") as f:
            station_info.to_csv(f, index=False, header=True)

        logger.info_green("Processing station distance data.")
        station_distance = self._process_distance(station_info=station_info)
        with open(self._distance_file, mode="w", encoding="utf-8", newline="") as f:
            station_distance.to_csv(f, index=False, header=True)


class WeatherPipeline(DataPipeline):
    """Generate weather data bin for the specified topology from frontierweather.com.

    Generated files will be generated in ~/.maro/data/citi_bike/[topology]/_build.
    Folder structure:
    ~/.maro
            /data/citi_bike/[topology]
                                    /_build bin data file
                                    /source
                                            /_download original data file
                                            /_clean cleaned data file
            /temp download temp file

    Args:
        topology(str): Topology name of the data file.
        source(str): Source url of original data file.
        is_temp(bool): (optional) If the data file is temporary.
    """

    _last_day_temp = None  # used to fill the temp for days which have no temp info

    _download_file_name = "weather.csv"

    _clean_file_name = "weather.csv"

    _build_file_name = "KNYC_daily.bin"

    _meta_file_name = "weather.yml"

    class WeatherEnum(Enum):
        SUNNY = 0
        RAINY = 1
        SNOWY = 2
        SLEET = 3

    def __init__(self, topology: str, source: str, is_temp: bool = False):
        super().__init__("citi_bike", topology, source, is_temp)

        self._common_data = {}

    def clean(self):
        """Clean the original data file."""
        super().clean()
        if os.path.exists(self._download_file):
            self._new_file_list.append(self._clean_file)
            logger.info_green("Cleaning weather data.")
            self._preprocess(input_file=self._download_file, output_file=self._clean_file)
        else:
            logger.warning(f"Not found downloaded weather data: {self._download_file}.")

    def _weather(self, row: dict):
        water_str = row["Precipitation Water Equiv"]
        water = round(float(water_str), 2) if water_str != "" else 0.0

        snow_str = row["Snowfall"]
        snow = round(float(snow_str), 2) if snow_str != "" else 0.0

        if snow > 0.0 and water > 0:
            return WeatherPipeline.WeatherEnum.SLEET.value
        elif water > 0.0:
            return WeatherPipeline.WeatherEnum.RAINY.value
        elif snow > 0.0:
            return WeatherPipeline.WeatherEnum.SNOWY.value
        else:
            return WeatherPipeline.WeatherEnum.SUNNY.value

    def _parse_date(self, row: dict):
        dstr = row.get("Date", None)

        return dstr

    def _parse_row(self, row: dict):

        date = self._parse_date(row=row)
        wh = self._weather(row=row)
        temp_str = row["Avg Temp"]

        temp = round(float(temp_str), 2) if temp_str != "" and temp_str is not None else self._last_day_temp

        self._last_day_temp = temp

        return {"date": date, "weather": wh, "temp": temp} if date is not None else None

    def _preprocess(self, input_file: str, output_file: str):
        data: list = None

        with open(input_file, "rt") as fp:
            reader = csv.DictReader(fp)

            data = [self._parse_row(row=row) for row in reader]

        data = filter(None, data)

        with open(output_file, "w+") as fp:
            writer = csv.DictWriter(fp, ["date", "weather", "temp"])

            writer.writeheader()
            writer.writerows(data)


class CitiBikeTopology(DataTopology):
    """Data topology for a predefined topology of citi_bike scenario.

    Args:
        topology(str): Topology name of the data file.
        trip_source(str): Original source url of citi_bike data.
        station_info(str): Current status station info of the stations.
        weather_source(str): Original source url of weather data.
        is_temp(bool): (optional) If the data file is temporary.
    """

    def __init__(
            self, topology: str, trip_source: str, station_info: str, weather_source: str, is_temp: bool = False):
        super().__init__()
        self._data_pipeline["trip"] = CitiBikePipeline(topology, trip_source, station_info, is_temp)
        self._data_pipeline["weather"] = NOAAWeatherPipeline(topology, weather_source, is_temp)
        self._is_temp = is_temp

    def __del__(self):
        if self._is_temp:
            self.remove()


class CitiBikeToyPipeline(DataPipeline):
    """Generate synthetic business events and station initialization distribution for Citi Bike scenario,
    from the predefined toy topologies.

    Folder structure:
    ~/.maro
            /data/citi_bike/[topology]
                                    /_build bin data file and other necessory files

    Args:
        start_time(str): Start time of the toy data.
        end_time(str): End time of the toy data.
        stations(list): List of stations info.
        trips(list): List of trips probability.
        topology(str): Topology name of the data files.
        is_temp(bool): (optional) If the data file is temporary.
    """

    _clean_file_name = "trips.csv"

    _build_file_name = "trips.bin"
    _station_meta_file_name = "station_meta.csv"
    _distance_file_name = "distance_adj.csv"

    _meta_file_name = "trips.yml"

    def __init__(
            self, start_time: str, end_time: str, stations: list, trips: list, topology: str, is_temp: bool = False):
        super().__init__("citi_bike", topology, "", is_temp)
        self._start_time = start_time
        self._end_time = end_time
        self._stations = stations
        self._trips = trips

        self._distance_file = os.path.join(self._build_folder, self._distance_file_name)
        self._station_meta_file = os.path.join(self._build_folder, self._station_meta_file_name)

    def download(self, is_force: bool):
        """Toy datapipeline not need download process."""
        pass

    def _station_dict_to_pd(self, station_dict):
        """Convert dictionary of station information to pd series."""
        return pd.Series(
            [
                station_dict["id"],
                station_dict["capacity"],
                station_dict["init"],
                station_dict["lat"],
                station_dict["lon"],
            ],
            index=["station_index", "capacity", "init", "latitude", "longitude"])

    def _gen_stations(self):
        """Generate station meta csv."""
        self._new_file_list.append(self._station_meta_file)

        stations = pd.Series(self._stations).apply(self._station_dict_to_pd)
        stations["station_index"] = pd.to_numeric(stations["station_index"], errors="coerce", downcast="integer")
        stations["station_id"] = pd.to_numeric(stations["station_index"], errors="coerce", downcast="integer")
        stations["capacity"] = pd.to_numeric(stations["capacity"], errors="coerce", downcast="integer")
        stations["init"] = pd.to_numeric(stations["init"], errors="coerce", downcast="integer")
        with open(self._station_meta_file, "w", encoding="utf-8", newline="") as f:
            stations.to_csv(f, index=False, header=True)

        return stations

    def _gen_trip(self, tick):
        """Generate trip record."""
        ret_list = []
        cur_probability = random.uniform(0, 1)
        for trip in self._trips:
            if trip["probability"] >= cur_probability:
                ret = {}
                ret["start_time"] = tick
                ret["start_station_id"] = trip["start_id"]
                ret["end_station_id"] = trip["end_id"]
                ret["start_station_index"] = trip["start_id"]
                ret["end_station_index"] = trip["end_id"]
                ret["duration"] = random.uniform(0, 120)
                ret_list.append(ret)
        return ret_list

    def _gen_trips(self):
        """Generate trip records csv files."""
        cur_tick = pd.to_datetime(self._start_time)
        end_tick = pd.to_datetime(self._end_time)

        trips = []
        while cur_tick < end_tick:
            new_trips = self._gen_trip(cur_tick)
            trips.extend(new_trips)
            cur_tick += pd.Timedelta(120, unit="second")

        trips_df = pd.DataFrame.from_dict(trips)

        trips_df["start_station_index"] = pd.to_numeric(
            trips_df["start_station_index"], errors="coerce", downcast="integer"
        )
        trips_df["end_station_index"] = pd.to_numeric(
            trips_df["end_station_index"], errors="coerce", downcast="integer"
        )
        self._new_file_list.append(self._clean_file)
        with open(self._clean_file, "w", encoding="utf-8", newline="") as f:
            trips_df.to_csv(f, index=False, header=True)

        return trips_df

    def _gen_distance(self, station_init: pd.DataFrame):
        """Generate distance metrix csv file."""
        distance_adj = pd.DataFrame(
            0,
            index=station_init["station_index"],
            columns=station_init["station_index"],
            dtype=np.float
        )
        look_up_df = station_init[["latitude", "longitude"]]
        distance_df = distance_adj.apply(lambda x: pd.DataFrame(x).apply(lambda y: geopy.distance.distance(
            (look_up_df.at[x.name, "latitude"], look_up_df.at[x.name, "longitude"]),
            (look_up_df.at[y.name, "latitude"], look_up_df.at[y.name, "longitude"])
        ).km, axis=1), axis=1)
        self._new_file_list.append(self._distance_file)
        with open(self._distance_file, "w", encoding="utf-8", newline="") as f:
            distance_df.to_csv(f, index=False, header=True)

        return distance_df

    def clean(self):
        """Clean the original data file."""
        logger.info_green(f"Generating trip data for topology {self._topology}.")
        super().clean()
        stations = self._gen_stations()
        self._gen_trips()
        self._gen_distance(stations)


class WeatherToyPipeline(WeatherPipeline):
    """Generate weather data bin for the specified topology from frontierweather.com.

    It will be generated in ~/.maro/data/citi_bike/[topology]/_build.
    Folder structure:
    ~/.maro
            /data/citi_bike/[topology]
                                    /_build bin data file
                                    /source
                                            /_download original data file
                                            /_clean cleaned data file
            /temp download temp file

    Args:
        topology(str): Topology name of the data file.
        start_time(str): Start time of the toy data.
        end_time(str): End time of the toy data.
        is_temp(bool): (optional) If the data file is temporary.
    """

    def __init__(self, topology: str, start_time: str, end_time: str, is_temp: bool = False):
        super().__init__(topology, "", is_temp)
        self._start_time = start_time
        self._end_time = end_time

    def download(self, is_force: bool):
        """Toy datapipeline not need download process."""
        pass

    def clean(self):
        """Clean the original data file."""
        logger.info_green("Cleaning weather data.")
        DataPipeline.clean(self)
        self._new_file_list.append(self._clean_file)
        self._preprocess(output_file=self._clean_file)

    def _weather(self):
        water = round(float(random.uniform(-1, 1)), 2)

        snow = round(float(random.uniform(-1, 1)), 2)

        if snow > 0.0 and water > 0:
            return WeatherPipeline.WeatherEnum.SLEET.value
        elif water > 0.0:
            return WeatherPipeline.WeatherEnum.RAINY.value
        elif snow > 0.0:
            return WeatherPipeline.WeatherEnum.SNOWY.value
        else:
            return WeatherPipeline.WeatherEnum.SUNNY.value

    def _gen_weather(self, tick):
        date = tick.strftime("%m/%d/%Y %H:%M:%S")
        wh = self._weather()
        temp = round(float(random.uniform(-1, 1) * 40), 2)

        return {"date": date, "weather": wh, "temp": temp}

    def _preprocess(self, output_file: str):
        data: list = []

        cur_tick = pd.to_datetime(self._start_time)
        end_tick = pd.to_datetime(self._end_time)

        while cur_tick <= end_tick:
            new_weather = self._gen_weather(cur_tick)
            data.append(new_weather)
            cur_tick += pd.Timedelta(1, unit="day")

        with open(output_file, "w+") as fp:
            writer = csv.DictWriter(fp, ["date", "weather", "temp"])

            writer.writeheader()
            writer.writerows(data)


class CitiBikeToyTopology(DataTopology):
    """Data topology for a predefined toy topology of citi_bike scenario.

    Args:
        topology(str): Topology name of the data file.
        config_path(str): Config file path of the topology.
        is_temp(bool): (optional) If the data file is temporary.
    """

    def __init__(self, topology: str, config_path: str, is_temp: bool = False):
        super().__init__()
        self._is_temp = is_temp
        if config_path.startswith("~"):
            config_path = os.path.expanduser(config_path)
        if os.path.exists(config_path):
            with open(config_path) as fp:
                cfg = safe_load(fp)
                self._data_pipeline["trip"] = CitiBikeToyPipeline(
                    start_time=cfg["start_time"],
                    end_time=cfg["end_time"],
                    stations=cfg["stations"],
                    trips=cfg["trips"],
                    topology=topology,
                    is_temp=is_temp
                )
                self._data_pipeline["weather"] = WeatherToyPipeline(
                    topology=topology,
                    start_time=cfg["start_time"],
                    end_time=cfg["end_time"],
                    is_temp=is_temp
                )
        else:
            logger.warning(f"Config file {config_path} for toy topology {topology} not found.")

    def download(self, is_force: bool = False):
        pass

    def __del__(self):
        if self._is_temp:
            self.remove()


class CitiBikeProcess:
    """Contains all predefined data topologies of citi_bike scenario.

    Args:
        is_temp(bool): (optional) If the data file is temporary.
    """

    meta_file_name = "source_urls.yml"
    meta_root = os.path.join(StaticParameter.data_root, "citi_bike/meta")

    def __init__(self, is_temp: bool = False):
        self.topologies = {}
        self.meta_root = os.path.expanduser(self.meta_root)
        self._meta_path = os.path.join(self.meta_root, self.meta_file_name)

        with open(self._meta_path) as fp:
            self._conf = safe_load(fp)
            for topology in self._conf["trips"].keys():
                if topology.startswith("toy"):
                    self.topologies[topology] = CitiBikeToyTopology(
                        topology=topology,
                        config_path=self._conf["trips"][topology]["toy_meta_path"],
                        is_temp=is_temp
                    )
                else:
                    self.topologies[topology] = CitiBikeTopology(
                        topology=topology,
                        trip_source=self._conf["trips"][topology]["trip_remote_url"],
                        station_info=self._conf["station_info"]["ny_station_info_url"],
                        weather_source=self._conf["weather"][topology]["noaa_weather_url"],
                        is_temp=is_temp
                    )


class NOAAWeatherPipeline(WeatherPipeline):
    """Generate weather data bin for the specified topology from ncei.noaa.gov.

    Generated files will be generated in ~/.maro/data/citi_bike/[topology]/_build.
    Folder structure:
    ~/.maro
            /data/citi_bike/[topology]
                                    /_build bin data file
                                    /source
                                            /_download original data file
                                            /_clean cleaned data file
            /temp download temp file

    Args:
        topology(str): Topology name of the data file.
        source(str): Source url of original data file.
        is_temp(bool): (optional) If the data file is temporary.
    """

    def __init__(self, topology: str, source: str, is_temp: bool = False):
        super().__init__(topology, source, is_temp)

    def download(self, is_force: bool):
        """Download the original data file."""
        super().download(is_force, self._gen_fall_back_file)

    def clean(self):
        """Clean the original data file."""
        DataPipeline.clean(self)
        if os.path.exists(self._download_file):
            self._new_file_list.append(self._clean_file)
            logger.info_green("Cleaning weather data.")
            self._preprocess(input_file=self._download_file, output_file=self._clean_file)
        else:
            logger.warning(f"Not found downloaded weather data: {self._download_file}.")

    def _weather(self, row):
        water = row["PRCP"] if row["PRCP"] is not None else 0.0

        snow = row["SNOW"] if row["SNOW"] is not None else 0.0

        if snow > 0.0 and water > 0:
            return WeatherPipeline.WeatherEnum.SLEET.value
        elif water > 0.0:
            return WeatherPipeline.WeatherEnum.RAINY.value
        elif snow > 0.0:
            return WeatherPipeline.WeatherEnum.SNOWY.value
        else:
            return WeatherPipeline.WeatherEnum.SUNNY.value

    def _preprocess(self, input_file: str, output_file: str):
        data: pd.DataFrame = pd.DataFrame()

        with open(input_file, "rt") as fp:
            org_data = pd.read_csv(fp)
            org_data["PRCP"] = pd.to_numeric(org_data["PRCP"], errors="coerce", downcast="integer")
            org_data["SNOW"] = pd.to_numeric(org_data["SNOW"], errors="coerce", downcast="integer")
            org_data["TMAX"] = pd.to_numeric(org_data["TMAX"], errors="coerce", downcast="integer")
            org_data["TMIN"] = pd.to_numeric(org_data["TMIN"], errors="coerce", downcast="integer")

            data["date"] = org_data["DATE"]
            data["weather"] = org_data.apply(self._weather, axis=1)
            data["temp"] = (org_data["TMAX"] + org_data["TMIN"]) / 2
        data.dropna(inplace=True)
        with open(output_file, mode="w", encoding="utf-8", newline="") as f:
            data.to_csv(f, index=False, header=True)

    def _gen_fall_back_file(self):
        fall_back_content = [
            "\"STATION\",\"DATE\",\"AWND\",\"PRCP\",\"SNOW\",\"TMAX\",\"TMIN\"\n",
            ",,,,,,\n"
        ]
        with open(self._download_file, mode="w", encoding="utf-8", newline="") as f:
            f.writelines(fall_back_content)
