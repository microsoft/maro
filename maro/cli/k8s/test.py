import os
import yaml

from maro.cli.k8s.aks_commands import add_job, init

from maro.utils.utils import LOCAL_MARO_ROOT 


SUBSCRIPTION = "03811e6d-e6a0-4ae1-92e7-2249b8cb13be"
RESOURCEGROUP = "yq"
DEPLOYMENT = "test"
LOCATION = "East US"

with open("test_conf.yml", 'r') as fp:
    conf = yaml.safe_load(fp)

init(conf)

# with open(os.path.join(LOCAL_MARO_ROOT, "examples", "rl", "config.yml")) as fp:
#     conf = yaml.safe_load(fp)

# add_job(conf)
