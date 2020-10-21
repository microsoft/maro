# Performance

Below are the final environment metrics of the onlineLP in different topologies.
For each experiment, we setup the environment and test for a duration of 1 week
with environment random seed 0, 128, 1024.  Besides the parameter listed in the
table, the experiment is configured with the default parameter value in the file
`examples/citi_bike/online_lp/config.yml`

## Future demand and supply from BE

Topology  | #Requirements | Shortage      | #Repositioning
----------|--------------:|--------------:|----------------:
toy.3s_4t |  15,071       |  8,168 +/-  7 |     173 +/-   4
toy.4s_4t |  10,128       |  4,061 +/-  5 |   1,628 +/-   8
toy.5s_6t |  15,983       |  6,562 +/- 17 |   2,505 +/-  15
ny.201908 | 371,969       | 14,197 +/- 67 | 123,295 +/- 287
ny.201910 | 351,855       | 11,348 +/- 46 | 104,165 +/- 582
ny.202001 | 169,304       |  1,900 +/- 36 |  71,236 +/- 162
ny.202004 |  91,810       |    154 +/-  2 |  18,239 +/- 208
ny.202006 | 197,833       |  1,899 +/- 21 |  36,515 +/- 208

## Future demand and supply by One-step Fixed-window Moving Average

Topology  | #Requirements | Shortage       | #Repositioning
----------|--------------:|---------------:|------------------:
toy.3s_4t |  15,071       |  8,167 +/-   1 |       1 +/-     0
toy.4s_4t |  10,128       |  3,711 +/-   3 |   2,462 +/-     6
toy.5s_6t |  15,983       |  6,628 +/-  16 |   2,194 +/-    20
ny.201908 | 371,969       | 12,616 +/- 113 | 277,553 +/- 1,331
ny.201910 | 351,855       |  8,705 +/- 132 | 291,526 +/- 1,557
ny.202001 | 169,304       |  1,901 +/-  37 |  71,236 +/-   162
ny.202004 |  91,810       |    108 +/-   1 | 114,066 +/-   941
ny.202006 | 197,833       |  1,080 +/-  21 | 186,164 +/- 1,613
