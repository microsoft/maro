# Rule Based Algorithm

## Overview

The VM scheduling problem is one of the quintessential use cases of the data center counts for the society. The rule based algorithm can be run with a set of scenario configurations. General experimental parameters (e.g., type of topology, type of algorithm to use) can be configured through `example/vm_scheduling/rule_based_algorithm/config.yml`.

## Algorithms

We offer some rule-based algorithms for VM scheduling problem, includings:

- **Random Pick**: When a VM request is coming, randomly choose an available PM with enough resources to place the VM.
- **First Fit**: When a VM request is coming, choose the available PM with enough resources whose index is smallest to place the VM.
- **Best Fit**: When a VM request is coming, choose the available PM with enough resources based on some rules to place the VM. There are four rules below:
  - Remaining CPU cores: Choose the available PM with minimal CPU cores.
  - Energy Consumption: Choose the available PM with maximum energy consumption. 
  - Remaining Memory: Choose the available PM with minimal memory.
  - Remaining CPU Cores and Energy Consumption: Choose the available PM with minimal CPU cores and maximum energy consumption.
- **Bin Packing**: Divide the PMs into several bins based on their remaining CPU cores. When a VM request is coming, allocate it to the bin, which can make the variance of the PM number in each bin as small as possible. If there are more than one PM in the chosen bin, randomly choose an available PM with enough resources.
- **Round Robin**: If the algorithm pick the PM $i$ at time $t−1$, then the algorithm will pick PM $i+1$ at current time $t$ for the coming VM request if PM $i+1$ is available, otherwise the algorithm will pick PM $i+2$ ... until the PM $i+k$ is available. 

## Metrics

We offer some metrics for evaluating the performance of each rule based algorithms, including:

- **Failed Allocation**: The number of the failed allocation VMs. 
- **Energy Consumption**: The total energy consumption of PMs. 

## Performance

We test the algorithm mentioned above on four topologies: **azure.2019.10k**, **azure.2019.10k.oversubscription**, **azure.2019.336k** and **azure.2019.336k.oversubscription**. The results are shown below (Noted that, for random pick, we try different random seed (666, 123) and average the results as the final result.)

The performance on topology **azure.2019.10k**

 #Algorithms  | #Failed Allocation | #Energy Consumption
:--------------:|:--------------:|:----------------:
Random Pick   | 135 |  2422294.29
FirstFit   | 0 |  2401318.95
Best Fit (Remaining CPU cores) | 0 | 2399610.25
Best Fit (Energy Consumption)  | 0 | 2398511.78
Best Fit (Remaining Memory)   | 0 | 2403865.39
Best Fit (Remaining CPU cores and Energy Consumption)   | 0 | 2399338.63
Bin Packing   | 56 | 2434414.40
Round Robin   | 72 | 2419616.66

The performance on topology **azure.2019.10k.oversubscription**

 #Algorithms  | #Failed Allocation | #Energy Consumption
:--------------:|:--------------:|:----------------:
Random Pick   | 117 |  2426808.51
FirstFit   | 0 |  2388954.69
Best Fit (Remaining CPU cores) | 0 | 2386371.94
Best Fit (Energy Consumption)  | 0 | 2385623.77
Best Fit (Remaining Memory)   | 0 | 2390528.03
Best Fit (Remaining CPU cores and Energy Consumption)   | 0 | 2386260.46
Bin Packing   | 26 | 2413872.27
Round Robin   | 72 | 2413679.75

The performance on topology **azure.2019.336k**

 #Algorithms  | #Failed Allocation | #Energy Consumption
:--------------:|:--------------:|:----------------:
Random Pick   | 225193 |  26452931.45
FirstFit   | 224625 | 26484605.11
Best Fit (Remaining CPU cores) | 226736 | 26425878.99
Best Fit (Energy Consumption)  | 224362 | 26489856.17
Best Fit (Remaining Memory)   | 224796 | 26493604.34
Best Fit (Remaining CPU cores and Energy Consumption)   | 228153 | 26431174.28
Bin Packing   | 248785 | 26284948.62
Round Robin   | 237510 | 26249634.61

The performance on topology **azure.2019.336k.oversubscription**

 #Algorithms  | #Failed Allocation | #Energy Consumption
:--------------:|:--------------:|:----------------:
Random Pick   | 225796 | 27449645.07
FirstFit   | 217709 | 27475405.93
Best Fit (Remaining CPU cores) | 220977 | 27440946.65
Best Fit (Energy Consumption)  | 218112 | 27475119.17
Best Fit (Remaining Memory)   | 218627 | 27479002.51
Best Fit (Remaining CPU cores and Energy Consumption)   | 221274 | 27413627.95
Bin Packing   | 317017 | 27675901.06
Round Robin   | 237849 | 27483855.78

## Run

1. Config
Modify some configuration about environment and algorithm accordingly in the config file `example/vm_scheduling/rule_based_algorithm/config.yml`.
2. Run
`python launcher.py`