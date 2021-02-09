# Rule Based Algorithm

## Overview

The VM scheduling problem is one of the quintessential use cases of MARO. The rule based algorithm can be run with a set of scenario configurations that can be found under maro/simulator/scenarios/vm_scheduling. General experimental parameters (e.g., type of topology, type of algorithm to use) can be configured through `config.yml`.

## Algorithms

- **Random Pick**: Random pick an available PM.
- **First Fit**: Choose the available PM whose index is smallest.
- **Best Fit**: Choose the available PMs based on some rules.
  - Remaining CPU cores: Choose the PM with minimal CPU cores.
  - Energy Consumption: Choose the PM with maximum energy consumption. 
  - Remaining Memory: Choose the PM with minimal memory.
  - Remaining CPU Cores and Energy Consumption: Choose the PM with minimal CPU cores and maximum energy consumption.
- **Bin Packing**: Divide the PMs into several bins, allocate the new VM in one of them, and let the variance of the PM number in each bin as small as possible. 
- **Round Robin**: If the algorithm pick the PM $i$ at time $t−1$, then the algorithm will pick PM $i+1$ at current time $t$ if PM $i+1$ is available, otherwise the algorithm will pick PM $i+2$ ... until the PM $i+k$ is available. 

## Metrics

- **Failed Allocation**: The number of the failed allocation VMs. 
- **Energy Consumption**: The total energy consumption of PMs. 

## Run

1. Config
Modify some configuration about environment and algorithm accordingly in the config file `config.yml`.
2. Run
`python launcher.py`