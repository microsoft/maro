Dashboard Visualization
=======================

Env-dashboard is a post-experiment visualization tool, aims to provide
more intuitive environment information, which will guide the design of
the algorithm and continually fine-tuning.

Dependency
----------

Module **streamlit** and **altair** should be pre-installed.

streamlit: An open-source app framework.

altair: A declarative statistical visualization library.

Start Dashboard
---------------

To start this visualization tool, maro should be pre-installed. The
command format is:

maro inspector env --source {Data\_Folder\_Path} --force {true/false}

e.g. maro inspector env --source ~\ *trade*\ 500 --force true

Expected data is dumped snapshot files. The structure of input file
folder should be like this:

--input\_file\_folder\_path 

----snapshot\_{int: epoch\_index} : folders to
restore data of each epoch 

--------holder\_info.csv: Attributes of current
epoch 

----manifest.yml: record basic info like scenario name, epoch\_num,
index\_name\_mapping file name. 

----index\_name\_mapping file: record the
relationship between an index and its name. type of this file varied
between scenario.

multiple summary files would be generated after data processing.

Usage
-----

Basically, each scenario has 2 parts of visualization according to the
overall and partial relationship. User could switch between them freely.

By changing sampling ratio and data display standard, user could view
the comprehensive and specific information by various types of charts
like line-chart, bar-chart or heat map.

When viewing data, users can interact freely, such as inputting custom
parameters according to predefined formulas, switching parameter
selection, etc.

Examples of each scenarios please refer to links below:

Citi Bike:

CIM:
