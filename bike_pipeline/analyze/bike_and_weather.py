import numpy as np
import pandas as pd
import json

def process_bike_data(bike_data_file):
    bike_data = None
    if os.path.exists(bike_data_file):
        with open(bike_data_file, mode="r", encoding="utf-8") as bike_csv_file:
            bike_data = pd.read_csv(bike_csv_file)
            bike_data['date'] = pd.to_datetime(bike_data['starttime']).dt.date
            bike_data['month'] = pd.to_datetime(bike_data['starttime']).dt.month
            bike_data['weekday'] = pd.to_datetime(bike_data['starttime']).dt.weekday
            bike_data['day'] = pd.to_datetime(bike_data['starttime']).dt.day
            bike_data['hour'] = pd.to_datetime(bike_data['starttime']).dt.hour
            print(bike_data)

            return bike_data
    return bike_data

def process_weather_data(weather_data_file):
    weather_data = None
    if os.path.exists(weather_data_file):
        with open(weather_data_file, mode="r", encoding="utf-8") as weather_csv_file:
            weather_data = pd.read_csv(weather_csv_file)
            weather_data['date'] = pd.to_datetime(weather_data['Date']).dt.date
    return weather_data

def merge_bike_weather(gp_bike_data, weather_data):
    combine_data = gp_bike_data.join(weather_data.set_index('date'), on='date')
    return combine_data

def process_station_data(station_data_file):
    station_data = None
    if os.path.exists(station_data_file):
        with open(station_data_file, mode="r", encoding="utf-8") as station_json_file:
            raw_station_data = pd.read_json(station_json_file)['features']
            station_data = raw_station_data.apply(_station_json_to_pd)
            #station_data['coordinates'] = station_data['features'].geometry.coordinates
            print(station_data)
    return station_data

def _gen_station_data(bike_data):
    gp_station_data_start = bike_data[['start station name','start station id', 'start station latitude', 'start station longitude']].drop_duplicates()
    gp_station_data_start.rename(columns={'start station name':'station_name','start station id':'station_id','start station latitude':'station_latitude', 'start station longitude':'station_longitude'},inplace=True)

    gp_station_data_end = bike_data[['end station name','end station id', 'end station latitude', 'end station longitude']].drop_duplicates()
    gp_station_data_end.rename(columns={'end station name':'station_name','end station id':'station_id','end station latitude':'station_latitude', 'end station longitude':'station_longitude'},inplace=True)

    station_data = pd.concat([gp_station_data_start, gp_station_data_end]).drop_duplicates().sort_values(by=['station_id'])
    print(station_data)
    return station_data

def _agg_bike_data(bike_data, columns):
    agg_data = bike_data.groupby(columns).size().reset_index()
    print(agg_data)
    return agg_data

def _station_json_to_pd(json_data):
    return pd.Series([json_data['geometry']['coordinates'][0], json_data['geometry']['coordinates'][1], json_data['properties']['station']['id'], json_data['properties']['station']['name'], json_data['properties']['station']['capacity']], index=['longitude', 'latitude', 'id', 'name', 'capacity'])

if __name__ == "__main__":
    #read bike data 
    import os
    import sys
    bike_data_file = sys.argv[1]
    weather_data_file = sys.argv[2]
    station_data_file = sys.argv[3]
    
    # bike_data = process_bike_data(bike_data_file)

    # station_data = _gen_station_data(bike_data)

    # from_to_all = _agg_bike_data(bike_data, ['start station id', 'end station id'])

    # from_all = _agg_bike_data(bike_data, ['start station id'])

    # to_all = _agg_bike_data(bike_data, ['end station id'])

    # trip_by_date = _agg_bike_data(bike_data, ['date'])

    # trip_by_weekday = _agg_bike_data(bike_data, ['weekday'])

    # trip_by_hour = _agg_bike_data(bike_data, ['hour'])

    # from_by_hour = _agg_bike_data(bike_data, ['start station id', 'hour'])

    # weather_data = process_weather_data(weather_data_file)

    # combine_data = merge_bike_weather(gp_bike_data, weather_data)

    sample_station_data = process_station_data(station_data_file)
    
