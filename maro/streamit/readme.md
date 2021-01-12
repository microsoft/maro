
Steps:

1. Start database service:

```bash

    cd streamit/server

    docker-compose up

```

2. Open database web console: http://localhost:9000

3. Send test data: python client.py

4. REST API for query: http://localhost:9000/exec?query=select * from <experiment name> where xxx, like following:

    http://localhost:9000/exec?query=select%20*%20from%20test_expmt_1610365168.2971027.port_detail

5. For more data, edit client.py or use web console to import, more on: https://questdb.io/docs/reference/client/web-console

6. About tables we will generate will sending data:

    . Create a maro.experiments table to save experiment information
    . Create a table for each category, named it as "<experiment name>.<category name>"
    . For each record in data table, we will append 3 additional fields: _tick, _episode, _ts.
      _tick and _episode are used as normal, but they are not indexed, _ts is composed with _episode and _tick, and used as fake timestamp for quick indexing. if need to query by episode and tick, use _ts with bit operation: episode << 16 | tick