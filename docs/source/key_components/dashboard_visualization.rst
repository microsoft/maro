Dashboard Visualization
=======================

Env-dashboard is a post-experimental visualization tool, aims to provide
more intuitive environment information, which will guide the design of
the algorithm and continually fine-tuning.

Currently, the visualization of senario **Container Inventory Management**
and **Citi Bike** are supported.

Dependency
----------

Module **streamlit** and **altair** should be pre-installed.

* `streamlit <https://www.streamlit.io/>`_: An open-source app framework.


* `altair <https://www.streamlit.io/>`_: A declarative statistical visualization library.

install them with:

.. code-block:: sh

    pip install streamlit altair

----

How to Use?
-----------

Get Data
~~~~~~~~

Experimental data is dumped automatically while running the experiment.
User need to run experiment through the file in folder maro/examples in source code. 
User could specify dump destination folder by setting the parameter **opts** as below:

.. code-block:: sh

    opts['enable-dump-snapshot'] = EXPECTED_OUTPUT_FOLDER

----

If user leave the parameter **opts** empty, data would be dumped to the folder containing 
experimental file by default.

Launch Visualization Tool
~~~~~~~~~~~~~~~~~~~~~~~~~

To start this visualization tool, user need to input command following the format:

.. code-block:: sh

    maro inspector env --source {source\_folder\_path} --force {true/false}

----

e.g.

.. code-block:: sh

    maro inspector env --source_path .\maro\dumper_files --force false

----

Parameter **force** refers to regenerate cross-epoch summary data or not, default value is 'true'.
Parameter **source_path** refers to the path of dumped snapshot files.
The expected structure of file folder should be like this:

Folder Structure

.. code-block:: sh

    ./LOCAL_DUMPER_DATA_FOLDER
        epoch_#                         # folders to restore data of each epoch.
            {resource_holder}.csv       # attributes of current epoch.
       manifest.yml                     # basic info like scenario name, number of epoches.
       index\_name\_mapping file        # relationship between an index and its name of resource holders.
       {resource_holder}_summary.csv    # cross-epoch summary information. 



----

If any file is missed compared with the expected folder structure
displayed above, the command line would prompt users with an error message.
The visualization tool looks for the free port to launch page in sequence, starting with port 8501.
The command line would print out the selected port.

Feature List
------------
Basically, each scenario has 2 parts of visualization: intra-epoch view
and inter-epoch view. User could switch between them freely.

Intra-epoch view
~~~~~~~~~~~~~~~~

User could view detailed information of selected resource holder or tick
under this mode. In order for users to better understand the data, we
separate the data into time dimension and space dimension. Users could view
both the value of a resource holder's property over time and the state of
all resource holders at a selected time (e.g. tick).

Content of intra-epoch view is varied between senarios. For example, in senario
container_inventory_management, the attributes of resource holders are relatively
complex. Thus, this view is divided into two parts: Accumulated Attributes and Detail Attributes.
The former one includes the heat map of transfer volume, top-k attributes summary,
accumulated attributes summary. The latter one includes the chart of two resource holders:
Port and Vessel attributes in the scenario container_inventory_management. 
Detailed introduction please refer to 
`Container Inventory Management Visualization <../scenarios/container_inventory_management.html#Visualization>`_.

The content of senario citi_Bike is much simpler,
mainly including top-k attributes summary and the chart of resource holder:
Station in senario citi_bike.
Detailed introduction please refer to 
`Citi Bike Visualization <../scenarios/citi_bike.html#Visualization>`_.

Epoch/Snapshot/Resource Holder Index Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To view the details of a resource holder or a tick, user could select
the specific index of epoch/snapshot/resource holder by sliding the slider
on the left side of page.

Snapshot/Resource Holder Sampling Ratio Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To view trends in the data, or to weed out excess information, user could
select the sampling ratio of snapshot/resource holder by sliding to
change the number of data to be displayed.

Formula Calculation
^^^^^^^^^^^^^^^^^^^

User could generate their own attributes by using pre-defined formulas.
The results of the formula calculation could be reused as the input
parameter of formula.


Inter-epoch view
~~~~~~~~~~~~~~~~

User could view cross-epoch information in this view.
In order to make users intuitively observe the results of the iterative
algorithm, such as whether the results converge as expected, we extracted
important attributes of resource holder from each epoch as a summary of
the current epoch and display them centrally in this view.
Users are free to choose the interval they care about and the sampling
rate within the selected interval. Line chart and bar chart can
effectively help users to know the results of the experiment.


Epoch Sampling Ratio Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To view trends in the data, or to weed out excess information, user could
select the sampling ratio of epoch by sliding to
change the number of data to be displayed.

Formula Calculation
^^^^^^^^^^^^^^^^^^^

Please refer to `Formula Calculation <#Feature List#Intra_epoch View#Formula Calculation>`_.


Examples
--------
Examples of each scenarios please refer to docs of each scenarios:

* `Container Inventory Management <../scenarios/container_inventory_management.html#Visualization>`_.

* `Citi Bike <../scenarios/citi_bike.html#Visualization>`_.
