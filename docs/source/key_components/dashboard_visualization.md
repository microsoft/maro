# Dashboard Visualization

Env-dashboard is a post-experiment visualization tool, aims to provide useful guidance for researchers and provide users with intuitive data feature display.

### Dependency

Module **streamlit** and **altair** should be pre-installed.

### start dashboard

To start this visualization tool, maro should be pre-installed. The command format is: 

maro inspector env --input {Data_Folder_Path}  --force {yes/no}

Expected data is dumped snapshot files. The structure of input file folder should be like this:

```python
--input_file_folder_path
    --snapshot_0 : data of each epoch
        --holder_info.csv: Attributes of current epoch
    ………………
    --snapshot_{epoch_num-1}
    --snapshot.manifest: record basic info like scenario name, index_name_mapping file name.
    --index_name_mapping file: record the relationship between an index and its name.
type of this file varied between scenario.
```

multiple summary files would be generated after data processing.

### dashboard examples

Since charts and front-end elements are varied between different scenarios. Examples would be provided separately.

##### citi_bike

In this scenario, only greedy policy is used. Since there is only 1 epoch, dashboard are divided into 2 parts: Summary & Detail.

###### Citi_Bike Summary Data

For summary data, user could view top-5 stations with different attributes.

![citi_bike_summary](..\images\visualization\dashboard\citi_bike_summary.gif)

###### Citi_Bike Detail Data

Detail data is divided into two dimensions according to time and space. 

If user choose to view information by station, it means that attributes of all snapshots within a selected station would be displayed.  By changing the option "station index", user could view data of different stations. By changing the option "Snapshot Sampling Ratio", Users can freely adjust the sampling rate. For example, if there are 100 snapshots and user selected 0.3 as sampling ratio, 30 snapshots data would be selected to render the chart.



![citi_bike_detail_by_station](..\images\visualization\dashboard\citi_bike_detail_by_station.gif)



To be specific, the line chart could be customized with operations in the following example.

By choosing the item "All", all of attributes would be displayed.  In addition, according to the data characteristics of each scenario, users will be provided with the option to quickly select a set of data.

e.g. In this scenario, item "Requirement Info"  refers to [trip_requirement, shortage, fulfillment].

![citi_bike_detail_by_station_2](..\images\visualization\dashboard\citi_bike_detail_by_station_2.gif)

Moreover, to improve the flexibility of visualizing data, user could use pre-defined formula and selected attributes to generate new attributes. Generated attributes would be treated in the same way as origin attributes.

![citi_bike_detail_by_station_3](..\images\visualization\dashboard\citi_bike_detail_by_station_3.gif)

If user choose to view information by snapshot, it means attributes of all holders within a selected snapshot would be displayed. By changing option "snapshot index", user could view data of different snapshot. By changing option "Snapshot Sampling Ratio", user could change the number of sampled data.

Particularly, if user want to check the name of a specific holder(station in this scenario), just hovering on the according bar.

Formula calculate please refer to [Citi_Bike Detail Data](#Citi_Bike Detail Data).

![citi_bike_detail_by_snapshot](..\images\visualization\dashboard\citi_bike_detail_by_snapshot.gif)

##### cim

In this scenario, the final result is generated through multiple rounds of iterative optimization. Basically, there are multiple {snapshot_(int)} folders under the user-input root path. The visualization is naturally more complicated. Generally speaking,  dumped data of this scenario is divided into two levels: Extro Epoch & Intra Epoch, refers to comparison of data between different epochs and display of data in one epoch respectively. 

###### Cim Extro Epoch Data

Generally speaking, the number of epochs in this scene will be greater than 100. In order to facilitate users to select specific data and observe the overall or partial data trend, the visualization tool provides data selection options in two dimensions.

To change "Start Epoch" and "End Epoch",  user could specify the selected data range. To change "Epoch Sampling Ratio", user could change the sampling rate of selected data, similar as [Citi_Bike Detail Data](#Citi_Bike Detail Data).

For description of Attributes Selection in charts and Formula Calculation, please refer to [Citi_Bike Detail Data](#Citi_Bike Detail Data).

![cim_extro_epoch](..\images\visualization\dashboard\cim_extro_epoch.gif)

###### Cim Intro Epoch Data

This part shows the data under a selected epoch. By scrolling the slider, users can select different epochs. Furthermore, this part of data is divided into two dimensions: by snapshot and by port according to time and space. In terms of data display, according to the different types of attributes, it is divided into two levels: acc data (accumulated attributes. e.g. acc_fulfillment) and detail data.

If user choose to view information by ports, attributes of the selected port would be displayed. 

Chart characteristics and data selection method please refer to [Citi_Bike Detail Data](#Citi_Bike Detail Data).

![cim_intra_epoch_by_ports](..\images\visualization\dashboard\cim_intra_epoch_by_ports.gif)

If user choose to view data by snapshots, attributes of selected snapshot would be displayed. The charts and data involved in this part are relatively rich, and we will introduce them by level.

**Acc Data**

This part includes the transfer volume hot map, bar chart of port accumulated attributes and top-5 ports of different attributes.

As shown in the following example, the x-axis and y-axis of transfer volume hot map refers to destinate port index and start port index respectively. The rect refers to the volume of cargoes transfer from start port to destinate port. By changing the snapshot index,  user could view the dynamic changes in the volume of cargo delivered by the port over time in the current epoch.

The bar chart of Port Acc Attributes displays the global change of ports.

![cim_intra_epoch_by_snapshot_acc_data](..\images\visualization\dashboard\cim_intra_epoch_by_snapshot_acc_data.gif)



**Detail Data**

Since the cargoes is transported through vessels, information of vessels could be viewed by snapshot.

Same as ports, user could change the sampling rate of vessels.

Bar chart of Port Detail Attributes and Formula Calculation please refer to [Citi_Bike Detail Data](#Citi_Bike Detail Data).

![cim_intra_epoch_by_snapshot_detail_data](..\images\visualization\dashboard\cim_intra_epoch_by_snapshot_detail_data.gif)

### Additional Information

##### Principle of Data Selection

In order to explain the data selection rules more intuitively, suppose the data are as follows:

Epoch_0: 

| snapshot_index | holder_index | attribute_1 |
| -------------- | ------------ | ----------- |
| 0              | 1            | 0-0-1       |
| 1              | 1            | 0-1-1       |
| 2              | 1            | 0-2-1       |
| 0              | 2            | 0-0-2       |
| 1              | 2            | 0-1-2       |
| 2              | 2            | 0-2-2       |

Epoch_1:

| snapshot_index | holder_index | attribute_1 |
| -------------- | ------------ | ----------- |
| 0              | 1            | 1-0-1       |
| 1              | 1            | 1-1-1       |
| 2              | 1            | 1-2-1       |
| 0              | 2            | 1-0-2       |
| 1              | 2            | 1-1-2       |
| 2              | 2            | 1-2-2       |

Data Selection:

1. Epoch 1. By holders. holder index=1 : [1-0-1, 1-1-1, 1-2-1]

2. Epoch 0. By snapshots. snapshot index=0: [0-0-0, 0-0-1,0-0-2]

    
