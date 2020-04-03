import os
import sys  
from h3 import h3

import json
import pandas as pd
from pandas.io.json import json_normalize


class hexagon():
    def __init__(self, filepath, savepath, scale, order):
        self.order = order
        self.scale = scale
        self.raw_data = self.read_csv_points(filepath)
        self.hexagon_data = self.hexagon_generater(self.raw_data)
        self.save_csv(self.hexagon_data, savepath)

    def hexagon_generater(self, df):
        # print(df)
        hexagon_df = df
        # hexagon_df = df.drop_duplicates(subset=["station_latitude","station_longitude"])
        # print(hexagon_df)
        hexagon_df = self.counts_by_hexagon(hexagon_df, self.scale)
        hexagon_df["neighbors"] = hexagon_df["hex_id"].apply(lambda x: h3.k_ring(x,self.order))
        return hexagon_df
    
    def get_neighbors(self, loc=None, hex_id=None):
        if loc:
            neighbor = self.hexagon_data[self.hexagon_data.geometry.coordinates == loc].neighbors.tolist()
        elif hex_id:
            neighbor = self.hexagon_data[self.hexagon_data.hex_id==hex_id].neighbors.tolist()
        else:
            return None
        
        return neighbor
    
    def get_hex_id(self, loc):
        hex_id = self.hexagon_data[(self.hexagon_data.geometry.coordinates == loc)].hex_id.tolist() if loc else None
        return hex_id
    
    def read_csv_points(self, filepath):
        
        '''Read a Dict objective into a dataframe and drop rows with null geometry. 
        Extract the station_latitude and station_longitude as separate columns from the geometry's coordinates'''
        
        df = pd.read_csv(filepath)
        # print(df)
        # df = df.drop_duplicates(subset=["station_latitude","station_longitude"])
        return df
    
    def save_csv(self, data, path):
        # print(data)
        data.to_csv(path)
    
    def read_dict_points(self, data_dict):
        
        '''Read a Dict objective into a dataframe and drop rows with null geometry. 
        Extract the station_latitude and station_longitude as separate columns from the geometry's coordinates'''
        
        df = pd.DataFrame.from_dict(data_dict,orient='id',columns='pd.DataFrame.from_dict')
        
        df['station_longitude'] = df["geometry.coordinates"].apply(lambda x: x[0]) 
        df['station_latitude'] = df["geometry.coordinates"].apply(lambda x: x[1]) 
        df = df.drop_duplicates(subset=["station_latitude","station_longitude"])

        return df
    
    def read_geojson_points(self, filepath):
        
        '''Read a GeoJSON file into a dataframe and drop rows with null geometry. 
        Extract the station_latitude and station_longitude as separate columns from the geometry's coordinates'''
        
        with open(filepath) as f:
            stops_geodata = json.load(f)
        
        df = pd.DataFrame(json_normalize(stops_geodata['features']))
        n_rows_orig = df.shape[0]
        
        df.dropna(subset=["geometry.coordinates"], inplace = True, axis = 0)
        n_rows_clean = df.shape[0]
        # print("Cleaning null geometries, eliminated ", n_rows_orig - n_rows_clean, 
        #     " rows out of the original ",n_rows_orig, " rows")
        
        df['station_longitude'] = df["geometry.coordinates"].apply(lambda x: x[0]) 
        df['station_latitude'] = df["geometry.coordinates"].apply(lambda x: x[1]) 
        df = df.drop_duplicates(subset=["station_latitude","station_longitude"])
        
        return df


    def counts_by_hexagon(self, df, resolution):
        
        '''Use h3.geo_to_h3 to index each data point into the spatial index of the specified resolution.
        Use h3.h3_to_geo_boundary to obtain the geometries of these hexagons'''

        # df = df[["station_latitude","station_longitude"]]
        
        df["hex_id"] = df.apply(lambda row: h3.geo_to_h3(row["station_latitude"], row["station_longitude"], resolution), axis = 1)
        
        df_aggreg = df.groupby(by = "hex_id").size().reset_index()
        df_aggreg.columns = ["hex_id", "station_num"]
        total_num = df_aggreg['station_num'].sum()
        avg_num = df_aggreg['station_num'].mean()
        # df_aggreg["geometry"] =  df_aggreg.hex_id.apply(lambda x: 
        #                                                     {    "type" : "Polygon",
        #                                                             "coordinates": 
        #                                                             [h3.h3_to_geo_boundary(h3_address=x,geo_json=True)]
        #                                                         }
        #                                                     )
        print(df_aggreg)
        print('avg station num in a hexagon: ', total_num, avg_num)
        return df

#filepath = "datasets_demo/busstops_Toulouse.geojson"
filepath = '/home/zhanyu/bikeData/ny/station/init/201306_202001.station.csv'
savepath = 'h3_' + filepath.split('/')[-1]
hexagon(filepath, savepath, 8, 1)