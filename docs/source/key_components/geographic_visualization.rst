Geographic Visualization
=======================

Env-geographic tool aims to provide users with an intuitive understanding
of Maro's experimental data and help users better understand Maro's scenarios and working processes.
Currently, Env-geographic supports two modes, which are real-time mode and local mode.
The former mode shows users real-time experimental data and helps users judge the effectiveness of the model.
The latter mode allows users to freely view the data after the experiment,
helping users to view the effect of the experiment and make subsequent decisions.


Dependency
----------

Env-geographic's startup depends on docker. 
Therefore, users need to install docker on the machine and ensure that it can run normally.
User could get docker through `Docker installation <https://docs.docker.com/get-docker/>`_.


How to Use?
-----------

Env-geographic has 3 parts: front-end, back-end and experiment database. To start this tool,
user need to start the database and service in order. The experimental data would
send to database automatically.

Start database
~~~~~~~~~~~~~
Firstly, user need to start the local database with command:

.. code-block:: sh

    maro inspector geo --start database

----

After the command is executed successfully, user
could view the front_end page through localhost:8080
and local data with localhost:9000 by default.
If the default port is occupied, user could obtain the access port of each container
through the following command:

.. code-block:: sh

    docker container ls

----

User could view all experiment information by SQL statement:

.. code-block:: SQL

    SELECT * FROM maro.experiments

----

Data is stored locally at the folder maro/maro/streamit/server/data.


Specify experiment name
~~~~~~~~~~~~~~~~~~~~~~~

To view the visualization of experimental data, user need to
specify the name of experiment. User could choose an existing
experiment or start an experiment either.

Choose an existing experiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

User could select a name from local database.

.. image:: ../images/visualization/geographic/database_exp.png
   :alt: database_exp

Start an experiment and send data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, users need to manually start the experiment to obtain
the data required by the service.

To send data to database, user need to set the value of the environment variable
"MARO_STREAMIT_ENABLED" to "true". If user wants to specify the experiment name,
set the environment variable "MARO_STREAMIT_EXPERIMENT_NAME". If user does not 
set this value, a unique experiment name would be processed automatically. User
could check the experiment name through database. It should be noted that when
selecting a topology, user must select a topology with specific geographic
information. The experimental data obtained by using topology files without
geographic information cannot be used in the Env-geographic tool.

.. code-block:: python

    os.environ["MARO_STREAMIT_ENABLED"] = "true"

    os.environ["MARO_STREAMIT_EXPERIMENT_NAME"] = "my_maro_experiment"

    # dump data to the folder which run the command.
    env = Env(scenario="cim", topology="global.22",
          start_tick=0, durations=100)

----

View the file maro/examples/hello_world/cim/hello.py to get complete reference.

After starting the experiment, make sure to query its name in local database.

To start the front-end and back-end service, user need to specify the experiment name
as following command:

.. code-block:: sh

    maro inspector geo --start service --experiment_name experiment_name.1614768800.6074605

----

The program will automatically determine whether to use real-time mode
or local mode according to the data status of the current experiment.

Feature List
------------

Real-time mode and local mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Local mode
^^^^^^^^^^

In this mode, user could comprehend the experimental data through the geographic
information and the charts on both sides. By clicking the play button in the lower
left corner of the page, user could view the dynamic changes of the data in the
selected time window. By hovering on geographic items and charts, more detailed information
could be displayed.

.. image:: ../images/visualization/geographic/local_mode.gif
   :alt: local_mode

The chart on the right side of the page shows the changes in the data over
a period of time from the perspectives of overall, port, and vessel.

.. image:: ../images/visualization/geographic/local_mode_right_chart.gif
   :alt: local_mode_right_chart

The chart on the left side of the page shows the ranking of the carrying
capacity of each port and the change in carrying capacity between ports
in the entire time window.

.. image:: ../images/visualization/geographic/local_mode_left_chart.gif
   :alt: local_mode_left_chart

Real-time mode
^^^^^^^^^^^^^^

The feature of real-time mode is not much different from that of local mode.
The particularity of real-time mode lies in the data. The automatic playback
speed of the progress bar in the front-end page is often close to the speed
of the experimental data. So user could not select the time window freely in
this mode.

.. image:: ../images/visualization/geographic/real_time_mode.gif
   :alt: real_time_mode

Geographic data display
~~~~~~~~~~~~~~~~~~~~~~~

In the map on the page, user can view the specific status of different resource
holders at various times. Users can further understand a specific area by zooming the map.
Among them, the three different status of the port:
Surplus, Deficit and Balance represent the quantitative relationship between the
empty container volume and the received order volume of the corresponding port
at that time.

.. image:: ../images/visualization/geographic/geographic_data_display.gif
   :alt: geographic_data_display

Data chart display
~~~~~~~~~~~~~~~~~~
The ranking table on the right side of the page shows the throughput of routes and
ports over a period of time. While the heat-map shows the throughput between ports
over a period of time. User can hover to specific elements to view data information.

The chart on the left shows the order volume and empty container information of each
port and each vessel. User can view the data of different resource holders by switching options.

In addition, user can zoom the chart to display information more clearly.

.. image:: ../images/visualization/geographic/data_chart_display.gif
   :alt: data_chart_display

Time window selection
~~~~~~~~~~~~~~~~~~~~~

This feature is only valid in local mode. User can select the starting point position by
sliding to select the left starting point of the time window, and view the specific data at
different time.

In addition, the user can freely choose the end of the time window. When the user plays this tool,
it will loop in the time window selected by the user.

.. image:: ../images/visualization/geographic/time_window_selection.gif
   :alt: time_window_selection


