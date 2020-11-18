Dashboard Visualization
=======================

Env-dashboard is a post-experiment visualization tool, aims to provide
more intuitive environment information, which will guide the design of
the algorithm and continually fine-tuning.

Feature List
------------
Basically, each scenario has 2 parts of visualization: intra-epoch view
and inter-epoch view. User could switch between them freely.

Intra-epoch view
~~~~~~~~~~~~~~~~
Detailed information of resource holders and selected tick could be
viewed under this mode.

Inter-epoch view
~~~~~~~~~~~~~~~~
Comprehensive and specific information by various types of charts like
line chart, bar chart or heat map could be viewed under this mode.

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
       |-- epoch_#                    # folders to restore data of
                                        each epoch
       |   |--{instance}_info.csv     # attributes of current epoch.
                                        Instance could be port,
                                        vessel or station
       |-- manifest.yml               # record basic info like
                                        scenario name, epoch\_num,
                                        index\_name\_mapping file name.
       |-- index\_name\_mapping file  # record the relationship between
                                        an index and its name.
                                        Type of this file varied
                                        between scenarios.
       |-- {instance}_summary.csv     # instance could be port,
                                        vessel or station.
                                        more detailed files,
                                        which will be used directly
                                        by the visualization tool.
                                        Generated after data processing.



----

Usage
-----

Basically, each scenario has 2 parts of visualization: inter epoch
and intra epoch. User could switch between them freely.

By changing sampling ratio and data display standard, user could view
the comprehensive and specific information by various types of charts
like line-chart, bar-chart or heat map.

When viewing data, users can interact freely, such as inputting custom
parameters according to predefined formulas, switching parameter
selection, etc.

Examples of each scenarios please refer to docs of each scenarios.
