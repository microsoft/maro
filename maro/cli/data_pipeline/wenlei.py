from maro.cli.data_pipeline.citi_bike import CitiBikeTopology


c1 = CitiBikeTopology( topology="ny.201801",trip_source="https://s3.amazonaws.com/tripdata/201801-citibike-tripdata.csv.zip",
                        station_info="https://gbfs.citibikenyc.com/gbfs/en/station_information.json",
                        weather_source="https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&dataTypes=PRCP,SNOW,TMAX,TMIN,AWND&stations=USW00094728&startDate=2016-01-01&endDate=2017-01-01&boundingBox=40.78,-74.0,40.76,-73.7",
                        is_temp=False)

# c1.download()
# c1._data_pipeline["trip"]._clean_file = "/home/maro/clustered_data/bin_ext/c1_2016/trips.csv"
# c1._data_pipeline["trip"]._station_meta_file = "/home/maro/clustered_data/bin_ext/c1_2016/station_meta.csv"
# c1._data_pipeline["trip"]._distance_file = "/home/maro/clustered_data/bin_ext/c1_2016/distance_adj.csv"
# c1._data_pipeline["trip"]._build_file = "/home/maro/clustered_data/bin_ext/c1_2016/trips.bin"
# c1._data_pipeline["trip"]._preprocess("/home/maro/clustered_data/c1_2016.csv")
# c1._data_pipeline["weather"].clean()
# c1._data_pipeline["weather"]._build_file = "/home/maro/clustered_data/bin_ext/c1_2016/KNYC_daily.bin"
# c1.build()

c1._data_pipeline["trip"]._clean_file = "/home/maro/clustered_data/bin_ext/c2_2016/trips.csv"
c1._data_pipeline["trip"]._station_meta_file = "/home/maro/clustered_data/bin_ext/c2_2016/station_meta.csv"
c1._data_pipeline["trip"]._distance_file = "/home/maro/clustered_data/bin_ext/c2_2016/distance_adj.csv"
c1._data_pipeline["trip"]._build_file = "/home/maro/clustered_data/bin_ext/c2_2016/trips.bin"
c1._data_pipeline["trip"]._preprocess("/home/maro/clustered_data/c2_2016.csv")
c1._data_pipeline["weather"]._build_file = "/home/maro/clustered_data/bin_ext/c2_2016/KNYC_daily.bin"
c1.build()

c1._data_pipeline["trip"]._clean_file = "/home/maro/clustered_data/bin_ext/c3_2016/trips.csv"
c1._data_pipeline["trip"]._station_meta_file = "/home/maro/clustered_data/bin_ext/c3_2016/station_meta.csv"
c1._data_pipeline["trip"]._distance_file = "/home/maro/clustered_data/bin_ext/c3_2016/distance_adj.csv"
c1._data_pipeline["trip"]._build_file = "/home/maro/clustered_data/bin_ext/c3_2016/trips.bin"
c1._data_pipeline["trip"]._preprocess("/home/maro/clustered_data/c3_2016.csv")
c1._data_pipeline["weather"]._build_file = "/home/maro/clustered_data/bin_ext/c3_2016/KNYC_daily.bin"
c1.build()

c1._data_pipeline["trip"]._clean_file = "/home/maro/clustered_data/bin_ext/c4_2016/trips.csv"
c1._data_pipeline["trip"]._station_meta_file = "/home/maro/clustered_data/bin_ext/c4_2016/station_meta.csv"
c1._data_pipeline["trip"]._distance_file = "/home/maro/clustered_data/bin_ext/c4_2016/distance_adj.csv"
c1._data_pipeline["trip"]._build_file = "/home/maro/clustered_data/bin_ext/c4_2016/trips.bin"
c1._data_pipeline["trip"]._preprocess("/home/maro/clustered_data/c4_2016.csv")
c1._data_pipeline["weather"]._build_file = "/home/maro/clustered_data/bin_ext/c4_2016/KNYC_daily.bin"
c1.build()

c1._data_pipeline["trip"]._clean_file = "/home/maro/clustered_data/bin_ext/regioned_2016/trips.csv"
c1._data_pipeline["trip"]._station_meta_file = "/home/maro/clustered_data/bin_ext/regioned_2016/station_meta.csv"
c1._data_pipeline["trip"]._distance_file = "/home/maro/clustered_data/bin_ext/regioned_2016/distance_adj.csv"
c1._data_pipeline["trip"]._build_file = "/home/maro/clustered_data/bin_ext/regioned_2016/trips.bin"
c1._data_pipeline["trip"]._preprocess("/home/maro/clustered_data/regioned_2016.csv")
c1._data_pipeline["weather"]._build_file = "/home/maro/clustered_data/bin_ext/regioned_2016/KNYC_daily.bin"
c1.build()