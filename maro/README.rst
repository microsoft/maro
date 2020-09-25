

.. image:: https://github.com/microsoft/maro/workflows/test/badge.svg
   :target: https://github.com/microsoft/maro/actions?query=workflow%3Atest
   :alt: test


.. image:: https://github.com/microsoft/maro/workflows/build/badge.svg
   :target: https://github.com/microsoft/maro/actions?query=workflow%3Abuild
   :alt: build


.. image:: https://github.com/microsoft/maro/workflows/docker/badge.svg
   :target: https://hub.docker.com/repository/docker/arthursjiang/maro
   :alt: docker


.. image:: https://readthedocs.org/projects/maro/badge/?version=latest
   :target: https://maro.readthedocs.io/
   :alt: docs



.. image:: https://raw.githubusercontent.com/microsoft/maro/master/docs/source/images/logo.svg
   :target: https://raw.githubusercontent.com/microsoft/maro/master/docs/source/images/logo.svg
   :alt: MARO LOGO

=======================================================================================================

Multi-Agent Resource Optimization (MARO) platform is an instance of Reinforcement
learning as a Service (RaaS) for real-world resource optimization. It can be
applied to many important industrial domains, such as container inventory
management in logistics, bike repositioning in transportation, virtual machine
provisioning in data centers, and asset management in finance. Besides
`Reinforcement Learning <https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf>`_ (RL),
it also supports other planning/decision mechanisms, such as
`Operations Research <https://en.wikipedia.org/wiki/Operations_research>`_.

Key Components of MARO:


* Simulation toolkit: it provides some predefined scenarios, and the reusable
  wheels for building new scenarios.
* RL toolkit: it provides a full-stack abstraction for RL, such as agent manager,
  agent, RL algorithms, learner, actor, and various shapers.
* Distributed toolkit: it provides distributed communication components, interface
  of user-defined functions for message auto-handling, cluster provision, and job orchestration.


.. image:: https://raw.githubusercontent.com/microsoft/maro/master/docs/source/images/maro_overview.svg
   :target: https://raw.githubusercontent.com/microsoft/maro/master/docs/source/images/maro_overview.svg
   :alt: MARO Key Components


Prerequisites
-------------

* `Python == 3.6/3.7 <https://www.python.org/downloads/>`_

Install MARO from Source (\ `Editable Mode <https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs>`_\ )
------------------------------------------------------------------------------------------------------------------------


* 
  Prerequisites


  * C++ Compiler

    * Linux or Mac OS X: ``gcc``
    * Windows: `Build Tools for Visual Studio 2017 <https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=15>`_ 

* 
  Enable Virtual Environment


  * 
    Mac OS / Linux

    .. code-block:: sh

       # If your environment is not clean, create a virtual environment firstly.
       python -m venv maro_venv
       source ./maro_venv/bin/activate

  * 
    Windows

    .. code-block:: powershell

       # If your environment is not clean, create a virtual environment firstly.
       python -m venv maro_venv
       .\maro_venv\Scripts\activate

       # You may need this for SecurityError in PowerShell.
       Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy Unrestricted

* 
  Install MARO


  * 
    Mac OS / Linux

    .. code-block:: sh

       # Install MARO from source.
       bash scripts/install_maro.sh

  * 
    Windows

    .. code-block:: powershell

       # Install MARO from source.
       .\scripts\install_maro.bat

Quick Example
-------------

.. code-block:: python

   from maro.simulator import Env

   env = Env(scenario="cim", topology="toy.5p_ssddd_l0.0", start_tick=0, durations=100)

   metrics, decision_event, is_done = env.step(None)

   while not is_done:
       metrics, decision_event, is_done = env.step(None)

   print(f"environment metrics: {env.metrics}")

Run Playground
--------------


* 
  Pull from `Docker Hub <https://hub.docker.com/repository/registry-1.docker.io/arthursjiang/maro/tags?page=1>`_

  .. code-block:: sh

     # Run playground container.
     # Redis commander (GUI for redis) -> http://127.0.0.1:40009
     # Local host docs -> http://127.0.0.1:40010
     # Jupyter lab with maro -> http://127.0.0.1:40011
     docker run -p 40009:40009 -p 40010:40010 -p 40011:40011 arthursjiang/maro:cpu

* 
  Build from source


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

    .. code-block:: powershell

       # Build playground image.
       .\scripts\build_playground.bat

       # Run playground container.
       # Redis commander (GUI for redis) -> http://127.0.0.1:40009
       # Local host docs -> http://127.0.0.1:40010
       # Jupyter lab with maro -> http://127.0.0.1:40011
       docker run -p 40009:40009 -p 40010:40010 -p 40011:40011 maro/playground:cpu

Contributing
------------

This project welcomes contributions and suggestions. Most contributions require
you to agree to a Contributor License Agreement (CLA) declaring that you have
the right to, and actually do, grant us the rights to use your contribution. For
details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether
you need to provide a CLA and decorate the PR appropriately (e.g., status check,
comment). Simply follow the instructions provided by the bot. You will only need
to do this once across all repos using our CLA.

This project has adopted the
`Microsoft Open Source Code of Conduct <https://opensource.microsoft.com/codeofconduct/>`_.
For more information see the
`Code of Conduct FAQ <https://opensource.microsoft.com/codeofconduct/faq/>`_
or contact `opencode@microsoft.com <mailto:opencode@microsoft.com>`_
with any additional questions or comments.

License
-------

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the `MIT <./LICENSE>`_ License.
