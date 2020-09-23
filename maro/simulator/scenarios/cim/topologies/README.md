# Topologies

- **toy.4p_ssdd** has 2 supply ports and 2 demand ports. Vessel capacities of this series of topologies are in {6000, 8000, 10000}, while the ratio of containers needed by order is sampled from [0.007, 0.009]
- **toy.5p_ssddd** has 2 supply ports and 3 demand ports. One of demand ports is a transport port at the same time. Vessel capacities of this series of topologies are in {8000, 9000, 10000}, while the ratio of containers needed by order is sampled from [0.018, 0.022]
- **toy.6p_sssbdd** has 3 supply ports, 2 demand ports and 1 balanced port. The balanced port and one of supply ports are 2 transport ports. Vessel capacities of this series of topologies are in {8000, 9000, 10000}, while the ratio of containers needed by order is sampled from [0.01, 0.02]
- **global_trade.22p** is a simplified global traffic topology.

## Grades of difficulty

- **l0.0** i.e. level 0.0, which is the simplest level without any fluctuation, noise or limitation.
- **l0.1** i.e. level 0.1, which has limited vessel capacity based on level 0.0.
- **l0.2** i.e. level 0.2, which has different capacity for different vessels based on level 0.1.
- **l0.3** i.e. level 0.3, which stipulated the number of orders subjecting to a sine function based on level 0.2.
- **l0.4** i.e. level 0.4, which adds noise while generating and allocating orders based on level 0.3.
- **l0.5** i.e. level 0.5, which adds noise on buffer ticks of shippers and consignees based on level 0.4.
- **l0.6** i.e. level 0.6, which adds noise on vessels' speed and parking duration based on level 0.5.
- **l0.7** i.e. level 0.7, which differs vessels' speed based on level 0.6.
- **l0.8** i.e. level 0.8, which is the hardest level in this release. Based on level 0.7, the number of orders subjecting to a function combined of two sine functions with different scopes and periods.
