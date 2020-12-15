# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.cli.data_pipeline.citi_bike import CitiBikeProcess
from maro.cli.data_pipeline.vm_scheduling import VmSchedulingProcess
from maro.utils.logger import CliLogger
from maro.utils.utils import deploy

logger = CliLogger(name=__name__)

scenario_map = {}
scenario_map["citi_bike"] = CitiBikeProcess
scenario_map["vm_scheduling"] = VmSchedulingProcess


def generate(scenario: str, topology: str = "", forced: bool = False, **kwargs):
    logger.info_green(
        f"Generating data files for scenario {scenario} topology {topology}"
        f" {'forced redownload.' if forced else ', not forced redownload.'}"
    )
    if scenario in scenario_map:
        process = scenario_map[scenario]()
        if topology in process.topologies:
            process.topologies[topology].download(forced)
            process.topologies[topology].clean()
            process.topologies[topology].build()
        else:
            logger.info_green(f"Please specify topology with -t in:{[x for x in process.topologies.keys()]}")
    else:
        logger.info_green(f"Please specify scenario with -s in:{[x for x in process.topologies.keys()]}")


def meta_deploy(*args, **kwargs):
    logger.info_green("Deploying data files for MARO to ~/.maro")
    deploy(False)


def list_env(*args, **kwargs):
    for scenario in scenario_map:
        process = scenario_map[scenario]()
        for topology in process.topologies:
            print(f"scenario: {scenario}, topology: {topology}")
