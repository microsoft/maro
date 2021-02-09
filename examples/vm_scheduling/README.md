# Simulation Results

Below table is the simulation results of current topologies based on `Best Fit` algorithm.

In the oversubscription topologies, the oversubscription rate is `115%`.

|Topology | PM Setting | Time Spent(s) | Total VM Requests |Successful Allocation| Energy Consumption| Total Oversubscriptions | Total Overload PMs
|:----:|-----|:--------:|:---:|:-------:|:----:|:---:|:---:|
|10k| 100 PMs, 32 Cores, 128 GB  | 104.98|10,000| 10,000| 2,399,610 | 0 | 0|
|10k.oversubscription| 100 PMs, 32 Cores, 128 GB|  101.00 |10,000 |10,000| 2,386,371| 279,331 | 0|
|336k| 880 PMs, 16 Cores, 112 GB | 7,896.37 |335,985| 109,249 |26,425,878 | 0 | 0 |
|336k.oversubscription| 880 PMs, 16 Cores, 112 GB | 7,903.33| 335,985| 115,008 | 27,440,946 | 3,868,475 | 0