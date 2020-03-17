import numpy as np
import pandas as pd

if __name__ == "__main__":
    #read bike data 
    import json
    import os
    import sys
    bike_data_file = sys.argv[1]
    weather_data_file = sys.argv[2]
    
    if os.path.exists(bike_data_file) and os.path.exists(weather_data_file):
        with open(bike_data_file, mode="r", encoding="utf-8") as bike_csv_file:
            bike_data = pd.read_csv(bike_csv_file)
            bike_data['date'] = pd.to_datetime(bike_data['starttime']).dt.date
            bike_data['hour'] = pd.to_datetime(bike_data['starttime']).dt.hour.floordiv(2)
            bike_data['weekday'] = pd.to_datetime(bike_data['starttime']).dt.weekday
            print(bike_data)
            gp_bike_data = bike_data.groupby(['start station name','date','hour']).count()
            print(gp_bike_data)
            #read weather data
            
            with open(weather_data_file, mode="r", encoding="utf-8") as weather_csv_file:
                weather_data = pd.read_csv(weather_csv_file)
                weather_data['date'] = pd.to_datetime(weather_data['Date']).dt.date
                print(weather_data)
                #merge data
                combine_data = gp_bike_data.join(weather_data.set_index('date'), on='date')
                print(combine_data)