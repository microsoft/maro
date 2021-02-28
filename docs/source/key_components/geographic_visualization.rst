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


How to Use?
-----------

Env-geographic has 3 parts: front-end, back-end and experiment database. To start this tool,
user need to start the docker containers, then start an experiment. The experimental data would
send to database automatically.

Start service
~~~~~~~~~~~~~
In order to start the env-geographic service, user need to start 4 docker
containers which are maro_vis_back_end_server, maro_vis_back_end_service,
maro_vis_front_end, questdb/questdbwith the following command:

.. code-block:: sh

    maro inspector geo

----

After the command is executed successfully, user
could view the front_end page through localhost:8080
and local data with localhost:9000.


Send experimental data
~~~~~~~~~~~~~~~~~~~~~~

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

.. code-block:: sh

    os.environ["MARO_STREAMIT_ENABLED"] = "true"

    os.environ["MARO_STREAMIT_EXPERIMENT_NAME"] = "my_maro_experiment"

    # dump data to the folder which run the command.
    env = Env(scenario="cim", topology="global.22",
          start_tick=0, durations=100)

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



.. image:: ../images/visualization/geographic/geographic_data_display.gif
   :alt: geographic_data_display

Data chart display
~~~~~~~~~~~~~~~~~~


Epoch Sampling Ratio Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To view trends in the data, or to weed out excess information, user could
select the sampling ratio of epoch by sliding to
change the number of data to be displayed.

.. image:: ../images/visualization/dashboard/epoch_sampling_ratio.gif
   :alt: epoch_sampling_ratio


Auxiliary options
~~~~~~~~~~~~~~~~~

Time window selection
^^^^^^^^^^^^^^^^^^^^^


Highlight route Selection
^^^^^^^^^^^^^^^^^^^^^^^^^


Epoch Selection
^^^^^^^^^^^^^^^

