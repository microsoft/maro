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
User could select the specific index of epoch/snapshot/resource holder
to view information.

Sampling Ratio
~~~~~~~~~~~~~~
User could select the sampling ratio of epoch/snapshot/resource holder
by sliding to change the number of data to be displayed.

Formula Calculation
~~~~~~~~~~~~~~~~~~~
User could generate their own attributes by using pre-defined formulas.

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

    maro inspector env --source .\maro\dumper_files --force true

----

Expected data is dumped snapshot files. The structure of input file
folder should be like this:

Folder Structure

.. code-block:: sh

   |-- ~/.source_folder_root
       |-- epoch_#                          # folders to restore data of
                                        each epoch
       |   |--{resource_holder}.csv     # attributes of current epoch.
                                        Instance could be port,
                                        vessel or station
       |-- manifest.yml                     # record basic info like
                                        scenario name, epoch\_num,
                                        index\_name\_mapping file name.
       |-- index\_name\_mapping file        # record the relationship
                                        between an index and its name.
                                        Type of this file varied
                                        between scenarios.
       |-- {resource_holder}_summary.csv    # instance could be port,
                                        vessel or station.
                                        more detailed files,
                                        which will be used directly
                                        by the visualization tool.
                                        Generated after data processing.



----

Examples
--------
Examples of each scenarios please refer to docs of each scenarios:
`Container Inventory Management <../scenarios/container_inventory_management.html#Visualization>`_.
`Citi Bike <../scenarios/citi_bike.html#Visualization>`_.
