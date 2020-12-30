Dashboard Visualization
=======================

Env-dashboard is a post-experiment visualization tool, aims to provide
more intuitive environment information, which will guide the design of
the algorithm and continually fine-tuning.

Currently, the visualization of senario Container Inventory Management
and Citi Bike are supported.

Feature List
------------
Basically, each scenario has 2 parts of visualization: intra-epoch view
and inter-epoch view. User could switch between them freely.

Intra-epoch view
~~~~~~~~~~~~~~~~

User could view detailed information of selected resource holders or tick
under this mode. In order for users to better understand the data, we
separate the data into time dimension and space dimension. Users could view
both the value of a resource holder's property over time and the state of
all resource holders at a selected time(e.g. tick).

Content of intra-epoch view is varied between senarios. For example, in senario
container_inventory_management, the attributes of resource holders are relatively
complex. Thus, this part is divided into two pieces: Accumulated Attributes and Detail Attributes.
The former includes the heat map of transfer volume, top-k attributes summary,
accumulated attributes summary. The latter includes the chart of two resource holders:
Port and Vessel attributes in the scenario container_inventory_management. The content of senario
citi_Bike is much simpler, mainly including top-k attributes summary and the chart of resource holder:
Station in this senario.

Inter-epoch view
~~~~~~~~~~~~~~~~

User could view cross-epoch information under this mode.
In order to make users intuitively observe the results of the iterative
algorithm, such as whether the results converge as expected, we extracted
important attributes of resource holder from each epoch as a summary of
the current epoch and display them centrally in the Inter-epoch View.
Users are free to choose the interval they care about and the sampling
rate within the selected interval. Line chart and bar chart can
effectively help users to know the results of the experiment.

Epoch/Snapshot/Resource Holder Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To view the details of a resource holder or a tick, user could select
the specific index of epoch/snapshot/resource holder by sliding the slider
on the left side of page.

Sampling Ratio
~~~~~~~~~~~~~~

To view trends in the data, or to weed out excess information, user could
select the sampling ratio of epoch/snapshot/resource holder by sliding to
change the number of data to be displayed.

Formula Calculation
~~~~~~~~~~~~~~~~~~~

User could generate their own attributes by using pre-defined formulas.
The results of the formula calculation could be reused as the input
parameter of formula.

Dependency
----------

Module **streamlit** and **altair** should be pre-installed.

--streamlit: An open-source app framework.

install it with:

.. code-block:: sh

    pip install streamlit

----

--altair: A declarative statistical visualization library.

install it with:

.. code-block:: sh

    pip install altair

----

How to Use?
-----------

To start this visualization tool, maro should be pre-installed. The
command format is:

.. code-block:: sh

    maro inspector env --source {source\_folder\_path} --force {true/false}

----

e.g.

.. code-block:: sh

    maro inspector env --source_path .\maro\dumper_files --force true

----

Parameter **force** refers to regenerate cross-epoch summary data or not, default value is 'true'.
Expected source_path is dumped snapshot files. The structure of input file
folder should be like this:

Folder Structure

.. code-block:: sh

    ~/.source_folder_root
        epoch_#                         # folders to restore data of each epoch.
            {resource_holder}.csv       # attributes of current epoch.
       manifest.yml                     # basic info like scenario name, epoch\_num.
       index\_name\_mapping file        # relationship between an index and its name of resource holders.
       {resource_holder}_summary.csv    # cross-epoch summary information. 



----

Check files in your folder carefully before launching the visualization tool.
If any file is missed compared with the expected folder structure
displayed above, the command line would prompt users with an error message.
The visualization Tool looks for the free port to launch page in sequence, starting with port 8501.

Examples
--------
Examples of each scenarios please refer to docs of each scenarios:
`Container Inventory Management <https://github.com/Meroy9819/maro/blob/v0.2_vis/docs/source/scenarios/container_inventory_management.rst#Visualization>`_.
`Citi Bike <https://github.com/Meroy9819/maro/blob/v0.2_vis/docs/source/scenarios/citi_bike.rst#Visualization>`_.
