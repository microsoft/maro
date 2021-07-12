# CIM: Using Real Data as the Input of the Simulation

Current the real data input are supported by the **CimRealDataLoader** and **CimRealDataContainer** implemented in *maro/data_lib/cim/cim_data_loader.py* and *maro/data_lib/cim/cim_data_container.py* respectively.

To build up a topology with real data as input, different to the synthetic mode, no need for any *config.yml* file,
just put the input data files under the folder of the specific topology.

## Input Data Files

The required input data files includes: *misc.yml, orders.csv, ports.csv, routes.csv, stops.csv, vessels.csv*.

The *misc.yml* file includes 7 settings, they are (for example):
```text
max_tick: 224
container_volume: 1
load_cost_factor: 0.05
dsch_cost_factor: 0.05
past_stop_number: 4
future_stop_number: 3
seed: 4096
```

The required header fields for the other files are listed below (*, more fields could be included if needed.*):

### ports.csv

index | name | capacity | empty | empty_return_buffer | empty_return_buffer_noise | full_return_buffer | full_return_buffer_noise
---|---|---|---|---|---|---|---

### routes.csv

index | name | port_name | distance_to_next_port
---|---|---|---

### vessels.csv

index | name | capacity | route_name | start_port_name | sailing_speed | sailing_speed_noise | parking_duration | parking_noise | empty
---|---|---|---|---|---|---|---|---|---

* *If the actual arrival_tick and departure_tick are provided, the vessels move according to the stops information (instead of the sailing_speed, sailing_speed_noise, parking_duration, parking_noise fields).*

* *But the sailing_speed and parking_duration will also be used to calculate the expected sailing period for each vessel on its route.*

### stops.csv

vessel_index | port_index | arrival_tick | departure_tick
---|---|---|---

### orders.csv

tick | source_port_index | dest_port_index | quantity
---|---|---|---

## Using MARO CLI to convert csv files into binary files

The CimRealDataLoader will read from the binary files instead of the csv files if there exists. Users can use the ```maro data build``` command to convert csv files to binary files which are smaller and also could be read faster. Examples:

```sh
maro data build --meta ~\.maro\data\cim\meta\cim.stops.meta.yml --file ~\.maro\data\cim\test.new_schema\stops.csv --output ~\.maro\data\cim\test.new_schema\stops.bin

maro data build --meta ~\.maro\data\cim\meta\cim.orders.meta.yml --file .~\.maro\data\cim\test.new_schema\orders.csv --output ~\.maro\data\cim\test.new_schema\orders.bin
```
