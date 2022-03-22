# Performance

Below are the final environment metrics of the greedy strategy in different topologies.
For each experiment, we setup the environment and test for a duration of 1 week
with environment random seed 0, 128, 1024.  Besides the parameter listed in the
table, the experiment is configured with the default parameter value in the file
`examples/citi_bike/greedy/config.yml`

Topology  | #Requirements | #Shortage     | #Repositioning
----------|--------------:|--------------:|----------------:
toy.3s_4t |  15,071       | 8,449 +/-  22 |  1,173 +/-  31
toy.4s_4t |  10,128       | 5,983 +/- 100 | 10,649 +/- 240
toy.5s_6t |  15,983       | 9,271 +/- 276 | 10,006 +/- 579
