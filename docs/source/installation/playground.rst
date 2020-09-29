
Playground Docker Image
=======================

Pull from `Docker Hub <https://hub.docker.com/repository/registry-1.docker.io/arthursjiang/maro/tags?page=1>`_
------------------------------------------------------------------------------------------------------------------

.. code-block:: sh

   # Run playground container.
   # Redis commander (GUI for redis) -> http://127.0.0.1:40009
   # Local host docs -> http://127.0.0.1:40010
   # Jupyter lab with maro -> http://127.0.0.1:40011
   docker run -p 40009:40009 -p 40010:40010 -p 40011:40011 arthursjiang/maro:cpu

Run from Source
---------------


* 
  Mac OS / Linux

  .. code-block:: sh

     # Build playground image.
     bash ./scripts/build_playground.sh

     # Run playground container.
     # Redis commander (GUI for redis) -> http://127.0.0.1:40009
     # Local host docs -> http://127.0.0.1:40010
     # Jupyter lab with maro -> http://127.0.0.1:40011
     docker run -p 40009:40009 -p 40010:40010 -p 40011:40011 maro/playground:cpu

* 
  Windows

  .. code-block::

     # Build playground image.
     .\scripts\build_playground.bat

     # Run playground container.
     # Redis commander (GUI for redis) -> http://127.0.0.1:40009
     # Local host docs -> http://127.0.0.1:40010
     # Jupyter lab with maro -> http://127.0.0.1:40011
     docker run -p 40009:40009 -p 40010:40010 -p 40011:40011 maro/playground:cpu

Major Services in Playground
----------------------------

.. list-table::
   :header-rows: 1

   * - Service
     - Description
     - Host
   * - ``Redis Commander``
     - Redis web GUI.
     - http://127.0.0.1:40009
   * - ``Read the Docs``
     - Local host docs.
     - http://127.0.0.1:40010
   * - ``Jupyter Lab``
     - Jupyter lab with MARO environment, examples, notebooks.
     - http://127.0.0.1:40011


*(If you use other port mapping, remember to change the port number.)*

Major Materials in Root Folder
------------------------------

.. list-table::
   :header-rows: 1

   * - Folder
     - Description
   * - ``examples``
     - Showcases of predefined scenarios.
   * - ``notebooks``
     - Quick-start tutorial.


*(Those not mentioned in the table can be ignored.)*
