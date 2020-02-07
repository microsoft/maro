# Dashboard

## About

Dashboard is made of a set of tools for visualizing statistics data in a RL train experiment.

We choosed influxdb to store the experiment data, and Grafana as front-end framework.

We supplied an easy way of starting the influxdb and Grafana services.

We implemented a dashboard class for uploading data to data base in maro.utils.dashboard. You can customize the class set base on the dashboard class.

We defined 3 Grafana dashboards which shows common experiment statistics data, experiment compare data and rank list for shortage.

We developed 2 Grafana panel plugins for you to customize your own dashboard in Grafana: Simple line chart, Heatmap chart. Simple line chart can show multiple lines in one chart with little setup. Heatmap chart can show z axis data as different red rects on different x, y axis values.

## Quick Start

- Start services

If you pip installed maro project, you need to make sure [docker](https://docs.docker.com/install/) is installed and create a folder for extracting dashboard resource files:

```shell
mkdir dashboard_services
cd dashboard_services
maro --dashboard
maro --dashboard start
```

If you start in source code of maro project, just cd maro/utils/dashboard/dashboard_resource

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

dashboard = DashboardBase('test_case_01', '.', True)
for i in range(10):
    fields = {'student_01':random.random()*10*i,'student_02':random.random()*15*i}
    tag = {'ep':i}
    measurement = 'score'
    dashboard.send(fields=fields,tag=tag,measurement=measurement)
```

- View the data chart in Grafana

Open url [http://localhost:50303](http://localhost:50303) in your browser.

Input user: admin and password: admin, skip the password change page if you wish to.

Then Grafana will navigate to 'Home' dashboard, tap 'Home' in up-left corner and select 'Hello World' option.

Grafan will navigate to 'Hello World' dashboard and the data chart panel will be shown in your browser.

## Deatil in Deploy

To make the Dashboard work, you need to setup the dockers for dashboard first, in a local machine or a remote one. And then you can insert the upload apis into the experiment, so the experiment data will be uploaded while the experiment running.

### Setup Services

- Install docker
- Prepare user can run docker
- Check out the socket ports for docker specified in dashboard_resource/docker-compose.yml are available, you can customize the ports if necessory
- Run dashboard_resource/start.sh with the user, the docker for influxdb and grafana will be started

```sh
cd dashboard_resource; bash start.sh
```

### Insert Upload Apis into experiment Code

- New a DashboardBase object with experiment name, log folder and log enabled
- Set the parameters for influxdb if necessory, it has 4 more parameters:
  
    host (str): influxdb ip address, default is localhost
    port (int): influxdb http port, default is 50301
    use_udp (bool): if use udp port to upload data to influxdb, default is true
    udp_port (int): influxdb udp port, default is 50304

```python
from maro.utils.dashboard import DashboardBase
dashboard = DashboardBase('test_case_01', '.', True)
```

#### Basic upload Api

the basic upload api is send()

```python
dashboard.send(fields={'port1':5,'port2':12}, tag={'ep':15}, measurement='shortage')
```

send() requires 3 parameters:

- fields ({Dict}): dictionary of fields, key is field name, value is field value, the data you want to draw in the dashboard charts.

    i.e.:{"port1":1024, "port2":2048}

- tag ({Dict}): dictionary of tag, used for query the specify data from database for the dashboard charts.

    i.e.:{"ep":5}

- measurement (string): type of fields, used as data table name in database.

    i.e.:"shortage"

#### Ranklist upload api

The ranklist upload api is upload_to_ranklist()

```python
dashboard.upload_to_ranklist(ranklist={'enabled':true, 'name':'test_shortage_ranklist'}, fields={'shortage':128})
```

upload_to_ranklist() require 2 parameters:

- ranklist ({Dict}): a ranklist dictionary, should contain "enabled" and "name" attributes
    i.e.:       {
                'enabled': True
                'name': 'test_shortage_ranklist'
                }

- fields ({Dict}): dictionary of field, key is field name, value is field value
    i.e.:{"train":1024, "test":2048}

#### Customized Upload Apis

The customized upload api includes upload_ep_data(), upload_shortage()..., they packed the basic upload api. The customized upload api required some business data, reorganized them into basic api parameters, and send data to database via basic upload api.

```python
from maro.utils.dashboard import DashboardBase

class DashboardECR(DashboardBase):
    def __init__(self, experiment: str, log_folder: str = None, host: str = 'localhost', port: int = 50301, use_udp: bool = True, udp_port: int = 50304):
        DashboardBase.__init__(self, experiment, log_folder, host, port, use_udp, udp_port)

    def upload_ep_data(self, fields, ep, measurement):
        fields['ep'] = ep
        self.send(fields=fields, tag={
            'experiment': self.experiment}, measurement=measurement)
```

upload_ep_data() requires 3 parameters:

- fields ({Dict}): dictionary of ep data, key is ep data name, value is ep data value.

    i.e.:{"port1":1024, "port2":2048}

- ep (int): current ep of the data, used as fields information to identify data of different ep in database.

- measurement (str): specify the measurement which the data will be stored in.

### Run Experiment

So that the experiment data is uploaded to the influxdb.

### View the Dashboards in Grafana

- Open Grafana link [http://localhost:50303](http://localhost:50303) (update the host and port if necessary) in the browser and log in with user "admin" password "admin" (change the username and password if necessary)

- Check the dashboards, you can switch between the predefined dashboards in the top left corner of the home page of Grafana.

  - The "ECR Experiment Metric Statistics" dashboard provid the  port shortage - ep chart, port loss - ep chart, port exploration - ep chart, port shortage pre ep chart, port q curve pre ep chart, laden transfer between ports pre ep chart. You can switch data between different experiments and episode of different charts in the selects at the top of dashboard

  - The "ECR Experiment Comparison" dashboard can compare a measurement of a port between 2 different experiments

  - The "ECR Shortage Ranklist" dashboard provid a demo rank list of test shortages

  - The "Hello World" dashboard is used to review data uploaded in Hello World section

- You can customize the dashboard reference to [https://grafana.com/docs/grafana/latest/](https://grafana.com/docs/grafana/latest/)