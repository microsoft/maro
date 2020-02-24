Dashboard
=========

About
-----

The Dashboard is made of a set of tools for visualizing key indicator
statistics data in an RL training experiment.

Quick Start
-----------

-  Start services

If user pip installed MARO project, they need to make sure
`docker <https://docs.docker.com/install/>`__ is installed and create a
folder for extracting dashboard resource files:

.. code:: shell

    mkdir dashboard_services
    cd dashboard_services
    # Extract the dashboard resource files to the current working directory with the "-e" option.
    maro dashboard -e
    # Start the dashboard services with the "-s" option.
    maro dashboard -s

If user start in source code of MARO project, just cd
maro/utils/dashboard/dashboard\_resource

.. code:: shell

    cd maro/utils/dashboard/dashboard_resource

and then run the start.sh in resource files:

.. code:: shell

    bash start.sh

-  Upload experiment data

Use maro.utils.dashboard.DashboardBase object to upload some simple
data.

.. code:: python

    from maro.utils.dashboard import DashboardBase
    import random

    dashboard = DashboardBase('hello_world', '.')
    for i in range(10):
        fields = {'student_01':random.random()*10*i,'student_02':random.random()*15*i}
        tag = {'ep':i}
        measurement = 'score'
        dashboard.send(fields=fields,tag=tag,measurement=measurement)

-  View the data chart in Grafana

Open URL http://localhost:50303 in the browser.

Use default user: 'admin' and password: 'admin' to login.

Then Grafana will navigate to the 'Home' dashboard, tap 'Home' in the
up-left corner and select the 'Hello World' option.

Grafana will navigate to the 'Hello World' dashboard and the data chart
panel will be shown in the browser.

Backend
-------

We use influxdb to store the experiment data and Grafana to visualize
the experiment data.

We use docker to setup the influxdb and Grafana services.

We implement a DashboardBase class for uploading data to the influxdb in
maro.utils.dashboard. Users can customize their class base on the
DashboardBase class.

Predefined Dashboard
--------------------

For the ECR scenario, We define 3 Grafana dashboards that show
experiments statistics data, experiments data comparison and a rank list
for experiments.

Customized Panels
-----------------

We develop 4 Grafana panel plugins for users to customize their
dashboard in Grafana: Simple line chart, Heatmap chart, stack bar chart,
and dot chart. The simple line chart can show multiple lines in one
chart. The heatmap chart can show z-axis data as different red
rectangles on different x, y-axis values. The stack bar chart can show
multiple bar series stacked together by the x-axis. The dot chart can
show multiple dot series in one chart.

Details of Deploying
--------------------

To make the Dashboard work, the users need to start the dockers for the
dashboard first, on a local machine or a remote one. And then they can
use the upload API to upload experiment data to the influxdb database.

Setup Services
~~~~~~~~~~~~~~

-  Install docker
-  The socket port for the influxdb and Grafana can be customized in
   dashboard\_resource/docker-compose.yml
-  Run dashboard\_resource/start.sh, the docker for influxdb and Grafana
   will be started

.. code:: sh

    cd dashboard_resource; bash start.sh

Send experiment data
~~~~~~~~~~~~~~~~~~~~

-  New a DashboardBase object with experiment name, log folder
-  Set the parameters for influxdb if necessary, it has 4 more optional
   parameters:

   host (str): influxdb IP address, default is localhost

   port (int): influxdb Http port, default is 50301

   use\_udp (bool): if use UDP port to upload data to influxdb, default
   is true

   udp\_port (int): influxdb UDP port, default is 50304

.. code:: python

    from maro.utils.dashboard import DashboardBase
    dashboard = DashboardBase(experiment='test_case_01', log_folder='.')

Basic upload Api
^^^^^^^^^^^^^^^^

the basic upload API: send()

.. code:: python

    dashboard.send(fields={'port1':5,'port2':12}, tag={'ep':15}, measurement='shortage')

send() requires 3 parameters :

-  fields ({Dict}): a dictionary of fields, the key is a field name,
   value is field value, the data user wants to draw in the dashboard
   charts.

   Reference to
   `field <https://docs.influxdata.com/influxdb/v1.7/concepts/key_concepts/>`__\ #field-key

   i.e.:{"port1":1024, "port2":2048}

-  tag ({Dict}): a dictionary of tag, used to query the specified data
   from the database for the dashboard charts.

   Reference to
   `tag <https://docs.influxdata.com/influxdb/v1.7/concepts/key_concepts/#tag-key>`__

   i.e.:{"ep":5}

-  measurement (string): type of fields, used as a data table name in
   the database.

   Reference to
   `measurement <https://docs.influxdata.com/influxdb/v1.7/concepts/key_concepts/#measurement>`__

   i.e.:"shortage"

Ranklist upload API
^^^^^^^^^^^^^^^^^^^

The rank list upload API is upload\_to\_ranklist()

.. code:: python

    dashboard.upload_to_ranklist(ranklist={'enabled':true, 'name':'test_shortage_ranklist'}, fields={'shortage':128})

upload\_to\_ranklist() require 2 parameters:

-  rank list (str): a rank list name, used as a measurement in influxdb

   i.e.: 'test\_shortage\_ranklist'

-  fields ({Dict}): a dictionary of field, the key is a field name,
   value is a field value

   i.e.:{"train":1024, "test":2048}

ECR scenario specific API
^^^^^^^^^^^^^^^^^^^^^^^^^

In the ECR scenario, the customized upload API includes
upload\_exp\_data(), packs the basic upload API. The customized upload
API requires some business data, reorganizes them into basic API
parameters, and sends data to the database via basic upload API.

.. code:: python


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

upload\_exp\_data() requires 4 parameters:

-  fields ({Dict}): dictionary of experiment data, key is experiment
   data name, value is experiment data value.

   i.e.:{"port1":1024, "port2":2048}

-  ep (int): current ep of the experiment data, used to identify data of
   different ep in the database.

-  tick (int): current tick of the experiment data, used to identify
   data of different ep in the database. Set None if it is not needed.

-  measurement (str): specify the measurement in which the data will be
   stored in.

Run Experiment
~~~~~~~~~~~~~~

So that the experiment data is uploaded to the influxdb.

View the Dashboards in Grafana
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Open Grafana link http://localhost:50303 (update the host and port if
   necessary) in the browser and log in with default user "admin"
   password "admin"

-  Check the dashboards, user can switch between the predefined
   dashboards in the top left corner of the home page of Grafana.

-  The "ECR Experiment Metric Statistics" dashboard provides the port
   shortage - ep chart, port loss - ep chart, port exploration - ep
   chart, port shortage pre ep chart, port q curve pre ep chart, laden
   transfer between ports pre ep chart. User can switch data between
   different experiments and an episode of different charts in the
   selects at the top of the dashboard

-  The "ECR Experiment Comparison" dashboard can compare the measurement
   of a port between 2 different experiments

-  The "ECR Shortage Ranklist" dashboard provides a demo rank list of
   test shortages

-  The "Hello World" dashboard is used to review data uploaded in Hello
   World section

-  User can customize the dashboard reference to
   https://grafana.com/docs/grafana/latest/


