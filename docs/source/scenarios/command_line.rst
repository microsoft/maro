Command support for scenarios
=================================

After installation, MARO provides a command that generate project for user,
make it much easier to use or customize scenario.


.. code-block:: sh

    maro project new

This command will show a step-by-step wizard to create a new project under current folder.
Currently it supports 2 modes.


1. Use built-in scenarios
-------------------------

To use built-in scenarios, please agree the first option "Use built-in scenario" with "yes" or "y", default is "yes".
Then you can select a built-in scenario and topologies with auto-completing.

.. code-block:: sh

    Use built-in scenario?yes
    Scenario name:cim
    Use built-in topology (configuration)?yes
    Topology name to use:global_trade.22p_l0.0
    Durations to emulate:1024
    Number of episodes to emulate:500
    {'durations': 1024,
    'scenario': 'cim',
    'topology': 'global_trade.22p_l0.0',
    'total_episodes': 500,
    'use_builtin_scenario': True,
    'use_builtin_topology': True}

    Is this OK?yes

If these settings correct, then this command will create a runner.py script, you can just run with:

.. code-block:: sh

    python runner.py

This script contains minimal code to interactive with environment without any action, you can then extend it as you wish.

Also you can create you own topology (configuration) if you say "no" for options "Use built-in topology (configuration)?".
It will ask you for a name of new topology, then copy the content from built-in one into your working folder (topologies/your_topology_name/config.yml).


2. Customized scenario
-------------------------------

This mode is used to generate a template of customize scenario for you instead of writing it from scratch.
To enable this, say "no" for option "Use built-in scenario", then provide your scenario name, default is current folder name.

.. code-block:: sh

    Use built-in scenario?no
    New scenario name:my_test
    New topology name:my_test
    Durations to emulate:1000
    Number of episodes to emulate:100
    {'durations': 1000,
    'scenario': 'my_test',
    'topology': 'my_test',
    'total_episodes': 100,
    'use_builtin_scenario': False,
    'use_builtin_topology': False}

    Is this OK?yes

This will generate following files like below:

.. code-block:: sh

    -- runner.py
    -- scenario
        -- business_engine.py
        -- common.py
        -- events.py
        -- frame_builder.py
        -- topologies
            -- my_test
                -- config.yml

The script "runner.py" is the entry of this project, it will interactive with your scenario without action.
Then you can fill "scenario/business_engine.py" with your own logic.