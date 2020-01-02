New Scenario
=============

If you need add a new scenario that using default Environment simulator :doc:`ECR scenario <../apidoc/maro.simulator>`,
you should get the source code, and then:

#. Create a folder for your new scenario under folder maro/simulator/scenarios, such as "dummy" for example.

#. Then add business_engine.py (NOTE: must be this name), and create a class that inherit from maro.simulaotor.simulator.AbsBusinessEngine.

#. Fill with your business logic

#. After completed your scenario, you can load with default Environment simulator like:

.. code-block:: python

    from maro.simulator import env

    env = env("your scenario", "your topology")