# Dashboard

## About

The Dashboard is made of a set of tools for visualizing statistics data in an RL train experiment.

We use influxdb to store the experiment data and Grafana as front-end framework.

We supply an easy way of starting the influxdb and Grafana services.

We implement a DashboardBase class for uploading data to the database in maro.utils.dashboard. You can customize your class base on the DashboardBase class.

We define 3 Grafana dashboards that show common experiment statistics data, experiments compare data and a rank list for experiments.

We develop 3 Grafana panel plugins for you to customize your dashboard in Grafana: Simple line chart, Heatmap chart, stack bar chart. The simple line chart can show multiple lines in one chart. The heatmap chart can show z-axis data as different red rectangles on different x, y-axis values. The stack bar chart can show multiple bar series stacked together by the x-axis.

## Quick Start

- Start services

If you pip installed MARO project, you need to make sure [docker](https://docs.docker.com/install/) is installed and create a folder for extracting dashboard resource files:

```shell
mkdir dashboard_services
cd dashboard_services
maro dashboard -e
maro dashboard -s
```

If you start in source code of MARO project, just cd maro/utils/dashboard/dashboard_resource

```shell
cd maro/utils/dashboard/dashboard_resource
```

and then run the start.sh in resource files:

```shell
bash start.sh
```

- Upload some data

Use maro.utils.dashboard.DashboardBase object to upload some simple data.

```python
from maro.utils.dashboard import DashboardBase
import random

dashboard = DashboardBase('hello_world', '.')
for i in range(10):
    fields = {'student_01':random.random()*10*i,'student_02':random.random()*15*i}
    tag = {'ep':i}
    measurement = 'score'
    dashboard.send(fields=fields,tag=tag,measurement=measurement)
```

- View the data chart in Grafana

Open URL [http://localhost:50303](http://localhost:50303) in your browser.

Use default user: 'admin' and password: 'admin' to login.

Then Grafana will navigate to the 'Home' dashboard, tap 'Home' in the up-left corner and select the 'Hello World' option.

Grafan will navigate to the 'Hello World' dashboard and the data chart panel will be shown in your browser.

## Detail in Deploy

To make the Dashboard work, you need to start the dockers for the dashboard first, in a local machine or a remote one. And then you can insert the upload API into the experiment process, so the experiment data will be uploaded while the experiment running.

### Setup Services

- Install docker
- Check out the socket ports for docker specified in dashboard_resource/docker-compose.yml are available, you can customize the ports if necessary
- Run dashboard_resource/start.sh, the docker for influxdb and Grafana will be started

```sh
cd dashboard_resource; bash start.sh
```

### Insert Upload Apis into experiment Code

- New a DashboardBase object with experiment name, log folder
- Set the parameters for influxdb if necessary, it has 4 more optional parameters:
  
    host (str): influxdb IP address, default is localhost

    port (int): influxdb Http port, default is 50301

    use_udp (bool): if use UDP port to upload data to influxdb, default is true

    udp_port (int): influxdb udp port, default is 50304

```python
from maro.utils.dashboard import DashboardBase
dashboard = DashboardBase('test_case_01', '.')
```

#### Basic upload Api

the basic upload API: send()

```python
dashboard.send(fields={'port1':5,'port2':12}, tag={'ep':15}, measurement='shortage')
```

send() requires 3 parameters (reference to [https://docs.influxdata.com/influxdb/v1.7/concepts/key_concepts/](https://docs.influxdata.com/influxdb/v1.7/concepts/key_concepts/)):

- fields ({Dict}): a dictionary of fields, the key is a field name, value is field value, the data you want to draw in the dashboard charts.

    Fields are a required piece of the InfluxDB data structure - you cannot have data in InfluxDB without fields.

    It’s also important to note that fields are not indexed.

    i.e.:{"port1":1024, "port2":2048}

- tag ({Dict}): a dictionary of tag, used to query the specified data from the database for the dashboard charts.

    Tags are optional. You don’t need to have tags in your data structure, but it’s generally a good idea to make use of them because, unlike fields, tags are indexed.

    This means that queries on tags are faster and that tags are ideal for storing commonly-queried metadata.

    i.e.:{"ep":5}

- measurement (string): type of fields, used as a data table name in the database.

    The measurement acts as a container for tags, fields, and the time column, and the measurement name is the description of the data that are stored in the associated fields.

    Measurement names are strings, and, for any SQL users out there, a measurement is conceptually similar to a table.

    i.e.:"shortage"

#### Ranklist upload API

The rank list upload API is upload_to_ranklist()

```python
dashboard.upload_to_ranklist(ranklist={'enabled':true, 'name':'test_shortage_ranklist'}, fields={'shortage':128})
```

upload_to_ranklist() require 2 parameters:

- rank list (str): a rank list name, used as a measurement in influxdb

    i.e.:       'test_shortage_ranklist'

- fields ({Dict}): a dictionary of field, the key is a field name, value is a field value

    i.e.:{"train":1024, "test":2048}

#### Customized Upload Apis

In the ECR example, the customized upload API includes upload_exp_data(), packs the basic upload API. The customized upload API requires some business data, reorganizes them into basic API parameters, and sends data to the database via basic upload API.

```python

from maro.utils.dashboard import DashboardBase

class DashboardECR(DashboardBase):
    def __init__(self, experiment: str, log_folder: str = None, host: str = 'localhost', port: int = 50301, use_udp: bool = True, udp_port: int = 50304):
        DashboardBase.__init__(self, experiment, log_folder, host, port, use_udp, udp_port)

    def upload_exp_data(self, fields, ep, tick, measurement):
        fields['ep'] = ep
        if tick is not None:
            fields['tick'] = tick
        self.send(fields=fields, tag={
            'experiment': self.experiment}, measurement=measurement)
```

upload_exp_data() requires 4 parameters:

- fields ({Dict}): dictionary of experiment data, key is experiment data name, value is experiment data value.

    i.e.:{"port1":1024, "port2":2048}

- ep (int): current ep of the experiment data, used to identify data of different ep in the database.

- tick (int): current tick of the experiment data, used to identify data of different ep in the database. Set None if it is not needed.

- measurement (str): specify the measurement in which the data will be stored in.

### Run Experiment

So that the experiment data is uploaded to the influxdb.

### View the Dashboards in Grafana

- Open Grafana link [http://localhost:50303](http://localhost:50303) (update the host and port if necessary) in the browser and log in with default user "admin" password "admin"

- Check the dashboards, you can switch between the predefined dashboards in the top left corner of the home page of Grafana.

  - The "ECR Experiment Metric Statistics" dashboard provides the port shortage - ep chart, port loss - ep chart, port exploration - ep chart, port shortage pre ep chart, port q curve pre ep chart, laden transfer between ports pre ep chart. You can switch data between different experiments and an episode of different charts in the selects at the top of the dashboard

  - The "ECR Experiment Comparison" dashboard can compare the measurement of a port between 2 different experiments

  - The "ECR Shortage Ranklist" dashboard provides a demo rank list of test shortages

  - The "Hello World" dashboard is used to review data uploaded in Hello World section

- You can customize the dashboard reference to [https://grafana.com/docs/grafana/latest/](https://grafana.com/docs/grafana/latest/)
