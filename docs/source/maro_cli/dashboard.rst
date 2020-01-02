Dashboard
=========

About
-----

Dashboard is made of a set of tools for visualizing statistics data in a
RL train experiment.

We choosed influxdb to store the experiment data, and Grafana as
front-end framework.

We supplied an easy way of starting the influxdb and Grafana services.

We implemented a dashboard class for uploading data to data base in
maro.utils.dashboard. You can customize the class set base on the
dashboard class.

We defined 3 Grafana dashboards which shows common experiment statistics
data, experiment compare data and rank list for shortage.

We developed 2 Grafana panel plugins for you to customize your own
dashboard in Grafana: Simple line chart, Heatmap chart. Simple line
chart can show multiple lines in one chart with little setup. Heatmap
chart can show z axis data as different red rects on different x, y axis
values.

Quick Start
-----------

-  Start services

If you pip installed maro project, you need to make sure
`docker <https://docs.docker.com/install/>`__ is installed and create a
folder for extracting dashboard resource files:

.. code:: shell

    mkdir dashboard_services
    cd dashboard_services
    maro --dashboard

If you start in source code of maro project, just cd
maro/utils/dashboard/dashboard\_resource

.. code:: shell

    cd maro/utils/dashboard/dashboard_resource

and then run the start.sh in resource files:

.. code:: shell

    bash start.sh

or use command "maro --dashboard start" to start the dashboard services.

-  Upload some data

Use maro.utils.dashboard.DashboardBase object to upload some simple
data.

.. code:: python

    from maro.utils.dashboard import DashboardBase
    import random

    dashboard = DashboardBase('test_case_01', '.')
    dashboard.setup_connection()
    for i in range(10):
        fields = {'student_01':random.random()*10*i,'student_02':random.random()*15*i}
        tag = {'ep':i}
        measurement = 'score'
        dashboard.send(fields=fields,tag=tag,measurement=measurement)

-  View the data chart in Grafana

Open url http://localhost:50303 in your browser.

Login with the default user: admin and password: admin, change the password if
you wish to.

Then Grafana will navigate to 'Home' dashboard, tap 'Home' in up-left
corner and select 'Hello World' option.

Grafan will navigate to 'Hello World' dashboard and the data chart panel
will be shown in your browser.

Deatil in Deploy
----------------

To make the Dashboard work, you need to setup the dockers for dashboard
first, in a local machine or a remote one. And then you can insert the
upload apis into the experiment, so the experiment data will be uploaded
while the experiment running.

Setup Services
~~~~~~~~~~~~~~

-  Install docker
-  Prepare user can run docker
-  Check out the socket ports for docker specified in
   dashboard\_resource/docker-compose.yml are available, you can
   customize the ports if necessory
-  Run dashboard\_resource/start.sh with the user, the docker for
   influxdb and grafana will be started

.. code:: sh

    cd dashboard_resource; bash start.sh

Insert Upload Apis into experiment Code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  New a DashboardBase object with experiment name
-  Make sure setup\_connection function of the object was called before
   send data.
-  Set the parameters of setup\_connection if necessory, the
   setup\_connection has 4 parameters:

   ::

                host (str): influxdb ip address
                port (int): influxdb http port
                use_udp (bool): if use udp port to upload data to influxdb
                udp_port (int): influxdb udp port

.. code:: python

    from maro.utils.dashboard import DashboardBase
    dashboard = DashboardBase('test_case_01', '.')
    dashboard.setup_connection()

Basic upload Api
^^^^^^^^^^^^^^^^

the basic upload api is send()

.. code:: python

    dashboard.send(fields={'port1':5,'port2':12}, tag={'ep':15}, measurement='shortage')

send() requires 3 parameters:

-  fields ({Dict}): dictionary of fields, key is field name, value is
   field value, the data you want to draw in the dashboard charts.

   i.e.:{"port1":1024, "port2":2048}

-  tag ({Dict}): dictionary of tag, used for query the specify data from
   database for the dashboard charts.

   i.e.:{"ep":5}

-  measurement (string): type of fields, used as data table name in
   database.

   i.e.:"shortage"

Ranklist upload api
^^^^^^^^^^^^^^^^^^^

The ranklist upload api is upload\_to\_ranklist()

.. code:: python

    dashboard.upload_to_ranklist(ranklist={'enabled':true, 'name':'test_shortage_ranklist'}, fields={'shortage':128})

upload\_to\_ranklist() require 2 parameters:

-  ranklist ({Dict}): a ranklist dictionary, should contain "enabled"
   and "name" attributes i.e.: { 'enabled': True 'name':
   'test\_shortage\_ranklist' }

-  fields ({Dict}): dictionary of field, key is field name, value is
   field value i.e.:{"train":1024, "test":2048}

Customized Upload Apis
^^^^^^^^^^^^^^^^^^^^^^

The customized upload api includes upload\_d\_error(),
upload\_shortage()..., they packed the basic upload api. The customized
upload api required some business data, reorganized them into basic api
parameters, and send data to database via basic upload api.

.. code:: python

    from maro.utils.dashboard import DashboardBase

    class DashboardECR(DashboardBase):
        def __init__(self, experiment: str, log_folder: str):
            DashboardBase.__init__(self, experiment, log_folder)

        def upload_shortage(self, nodes_shortage, ep):
            nodes_shortage['ep'] = ep
            self.send(fields=nodes_shortage, tag={
                'experiment': self.experiment}, measurement='shortage')

upload\_shortage() requires 2 parameters:

-  nodes\_shortage ({Dict}): dictionary of shortage of different nodes,
   key is node name, value is shortage value.

   i.e.:{"port1":1024, "port2":2048}

-  ep (number): current ep, used as x axis data in dashboard charts.

Run Experiment
~~~~~~~~~~~~~~

So that the experiment data is uploaded to the influxdb.

View the Dashboards in Grafana
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Open Grafana link http://localhost:50303 (update the host and port if
   necessary) in the browser and log in with user "admin" password
   "admin" (change the username and password if necessary)

-  Check the dashboards, you can switch between the predefined
   dashboards in the top left corner of the home page of Grafana.

   -  The "Experiment Metric Statistics" dashboard provid the port
      shortage - ep chart, port loss - ep chart, port exploration - ep
      chart, port shortage pre ep chart, port q curve pre ep chart,
      laden transfer between ports pre ep chart. You can switch data
      between different experiments and episode of different charts in
      the selects at the top of dashboard
   -  The "Experiment Comparison" dashboard can compare a measurement of
      a port between 2 different experiments
   -  The "Shortage Ranklist" dashboard provid a demo rank list of test
      shortages
   -  The "Hello World" dashboard is used to review data uploaded in
      Hello World section

-  You can customize the dashboard reference to
   https://grafana.com/docs/grafana/latest/


