Maro Commandline Interface
===================================

If you install maro by pip, you can use maro command:

Currently we have 2 commands:

#. maro --envs

    This command will list show available env configurations in package.

#. maro --dashboard

    This command will default extract dashboard resources files to current directory.

    Then you can use --dashboard start to start the Dashboard services, -- dashboard build to rebuild docker image for Dashboard services, or --dashboard stop to stop them.

    Read more about the Dashboard for MARO:  :doc:`Dashboard <./dashboard>`
