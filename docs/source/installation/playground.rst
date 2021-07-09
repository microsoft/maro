
Playground Docker Image
=======================

Pull from `Docker Hub <https://hub.docker.com/repository/registry-1.docker.io/maro2020/playground/tags?page=1>`_
------------------------------------------------------------------------------------------------------------------

.. code-block:: sh

   # Run playground container.
   # Redis commander (GUI for redis) -> http://127.0.0.1:40009
   # Jupyter lab with maro -> http://127.0.0.1:40010
   docker run -p 40009:40009 -p 40010:40010 maro2020/playground

Run from Source
---------------

* Mac OS / Linux

  .. code-block:: sh

     # Build playground image.
     bash ./scripts/build_playground.sh

     # Run playground container.
     # Redis commander (GUI for redis) -> http://127.0.0.1:40009
     # Jupyter lab with maro -> http://127.0.0.1:40010
     docker run -p 40009:40009 -p 40010:40010 maro2020/playground

* Windows

  .. code-block::

     # Build playground image.
     .\scripts\build_playground.bat

     # Run playground container.
     # Redis commander (GUI for redis) -> http://127.0.0.1:40009
     # Jupyter lab with maro -> http://127.0.0.1:40010
     docker run -p 40009:40009 -p 40010:40010 maro2020/playground

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
   * - ``Jupyter Lab``
     - Jupyter lab with MARO environment, examples, notebooks.
     - http://127.0.0.1:40010


*(Remember to change ports if you use different ports mapping.)*

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


*(The ones not mentioned in this table can be ignored.)*
