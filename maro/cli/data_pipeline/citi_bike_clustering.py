import os
from collections import defaultdict

import numpy as np
import pandas as pd
from pandas import DataFrame

from maro.cli.data_pipeline.utils import download_file

def download_2016files(output_pth):
    for i in range(1, 13):
        download_file(source=f"https://s3.amazonaws.com/tripdata/2016{i:02}-citibike-tripdata.zip", 
            destination=os.path.join(output_pth, f"2016{i:02}"))

def concat_dataframes(file_folder):
    file_pths = sorted([os.path.join(file_folder, f) for f in os.listdir(file_folder)])
    dataframes = [pd.read_csv(f) for f in file_pths]
    columns = list(dataframes[0].columns)
    tot_dataframe = pd.concat([df.set_axis(columns, axis='columns', inplace=False) for df in dataframes])
    return tot_dataframe

def station_mapping(trip_data:DataFrame, region_info:DataFrame):
    stations = trip_data.drop_duplicates(subset=["start station id", "start station latitude", 
        "start station longitude"])[["start station id", "start station latitude", "start station longitude"]]
    stations_np = np.array(stations)
    region_info_np = np.array(region_info)
    id_mapping = {}
    region_mapping = {}
    for row in region_info_np:
        inner_id = int(row[0])
        la = row[1]
        lg = row[2]
        for r1 in stations_np:
            la1 = r1[1]
            lg1 = r1[2]
            if abs(la1 - la) < 0.000001 and abs(lg1 - lg) < 0.000001:
                if inner_id in id_mapping:
                    print("conflicts!")
                id_mapping[inner_id] = int(r1[0]) 
                region_mapping[int(r1[0])] = {"capacity": int(row[3]), "region": int(row[4]), "latitude": la, 
                    "longitude": lg}
    return region_mapping

def region2cluster(cluster_info:DataFrame):
    cluster_np = np.array(cluster_info)
    return {r[0]: r[1:] for r in cluster_np}

def regionize_trip(trip:DataFrame, region_mapping:dict, cluster_mapping:dict):
    id_mapping = {k: int(v['region']) for k, v in region_mapping.items()}
    ids = set(id_mapping.keys())
    filtered_trip = trip[trip['start station id'].isin(ids) & trip['end station id'].isin(ids)]
    regioned_trip = filtered_trip.replace({"start station id": id_mapping, "end station id": id_mapping})

    for k, v in cluster_mapping.items():
        regioned_trip.loc[regioned_trip["start station id"] == k, "start station latitude"] = v[0]
        regioned_trip.loc[regioned_trip["end station id"] == k, "end station latitude"] = v[0]
        regioned_trip.loc[regioned_trip["start station id"] == k, "start station longitude"] = v[1]
        regioned_trip.loc[regioned_trip["end station id"] == k, "end station longitude"] = v[1]
        regioned_trip.loc[regioned_trip["start station id"] == k, "start station cluster"] = int(v[2])
        regioned_trip.loc[regioned_trip["end station id"] == k, "end station cluster"] = int(v[2])

    return regioned_trip

def separate_by_cluster(regioned_trip:DataFrame):
    cluster_ids = regioned_trip["start station cluster"].unique()
    inner_cluster = {}
    for id in cluster_ids:
        inner_cluster[int(id)] = regioned_trip[(regioned_trip["start station cluster"] == id) & 
            (regioned_trip["end station cluster"] == id)]
    inner_cluster[0] = regioned_trip
    return inner_cluster

def get_station_info(region_info:DataFrame, cluster_info:DataFrame, init_scale=3.5):
    region_np = np.array(region_info)
    cluster_np = np.array(cluster_info)
    capacity = defaultdict(int)
    for r in region_np:
        capacity[int(r[4])] += int(r[3])
    capacity_np = np.array([capacity[int(c[0])] for c in cluster_np])
    stations = DataFrame()
    stations["station_id"] = np.arange(capacity_np.shape[0])
    stations["station_index"] = cluster_info["index"]
    stations["capacity"] = capacity_np 
    stations["init"] = capacity_np//init_scale
    stations["latitude"] = cluster_info["start.station.latitude"]
    stations["longitude"] = cluster_info["start.station.longitude"]
    return stations

if __name__ == "__main__":
    download_pth = r"/data/test"
    # download_2016files(download_pth)
    trip_dataframe = concat_dataframes(download_pth)
    region_info = pd.read_csv(r"maro/cli/data_pipeline/citibike_cluster.txt")
    cluster_info = pd.read_csv(r'maro/cli/data_pipeline/citibike/citibike_region.txt')

    region_station_info = get_station_info(region_info, cluster_info)
    region_station_info.to_csv(r"/D_data/citibike/station_meta.csv")

    region_mapping = station_mapping(trip_dataframe, region_info)
    cluster_mapping = region2cluster(cluster_info)

    regioned_trip = regionize_trip(trip_dataframe, region_mapping=region_mapping, cluster_mapping=cluster_mapping)
    regioned_trip.to_csv(r"/D_data/citibike/regional_trip.csv")

    clusters = separate_by_cluster(regioned_trip)
    for k, v in clusters.items():
        v.to_csv(os.path.join(r"/D_data/citibike/", f"cluster_{k}.csv"))
    
    
    