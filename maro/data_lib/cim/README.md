# CIM: Using Real Data as the Input of the Simulation

Current the real data input are supported by the **CimRealDataLoader** and **CimRealDataContainer** implemented in *maro/data_lib/cim/cim_real_data_loader.py* and *maro/data_lib/cim/cim_real_data_container.py* respectively.

## Configurations in the Topology File *config.yml*

To build up a topology with real data as input, 2 parts of configurations are required for the config.yml:

- input_setting: describes what type of the input data files and where are the data files.
- transfer_cost_factors: used by the business engine to calculate the transfer cost automatically.

Here is an example for the *config.yml* file:

```text
input_setting:
  from_files: true
  input_type: real  # real, dumped
  data_folder: "~\.maro\data\cim\test.new_schema"

transfer_cost_factors:
  load: 0.05
  dsch: 0.05
```

## Input CSV Files

The required input data files includes: *misc.yml, orders.csv, ports.csv, routes.csv, stops.csv, vessels.csv*.

The *misc.yml* file includes 4 settings, they are (for example):
```text
max_tick: 224
container_volume: 1
past_stop_number: 4
future_stop_number: 3
```

The required header fields for the other files are listed below (*, more fields could be included if needed.*):

### ports.csv

index | name | capacity | empty | empty_return_buffer | empty_return_buffer_noise | full_return_buffer | full_return_buffer_noise
---|---|---|---|---|---|---|---

### routes.csv

index | name | port_name | distance
---|---|---|---

### vessels.csv

index | name | capacity | route_name | start_port_name | sailing_speed | sailing_speed_noise | parking_duration | parking_noise | period | empty
---|---|---|---|---|---|---|---|---|---|---

### stops.csv

vessel_index | port_index | arrive_tick | departure_tick
---|---|---|---

### orders.csv

tick | source_port_index | dest_port_index | quantity
---|---|---|---
