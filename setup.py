# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

######################################################################################
pk_description = """
Multi-Agent Resource Optimization (MARO) platform is an instance of Reinforcement
learning as a Service (RaaS) for real-world resource optimization.
"""

with open("./maro/README.rst", "r") as f:
    pk_long_description = f.read()

with open("LICENSE", "r") as f:
    pk_license = f.read()
######################################################################################

from glob import glob
import os
from setuptools import setup, find_packages, Extension
from setuptools.command.install import install
from setuptools.command.develop import develop
import sys

# Set environment variable to skip deployment process of MARO
os.environ["SKIP_DEPLOYMENT"] = "TRUE"

from maro import __version__

# root path to backend
BASE_SRC_PATH = "./maro/backends"
# backend module name
BASE_MODULE_NAME = "maro.backends"

# extensions to be compiled
extensions = []
cython_directives = {"embedsignature": True}
compile_conditions = {}

# CURRENTLY we using environment variables to specified compiling conditions
# TODO: used command line arguments instead

# specified frame backend
FRAME_BACKEND = os.environ.get("FRAME_BACKEND", "NUMPY")  # NUMPY or empty


# include dirs for frame and its backend
include_dirs = []

# backend base extensions
extensions.append(
    Extension(
        f"{BASE_MODULE_NAME}.backend",
        sources = [f"{BASE_SRC_PATH}/backend.c"])
)

if FRAME_BACKEND == "NUMPY":
    import numpy

    include_dirs.append(numpy.get_include())

    extensions.append(
        Extension(
            f"{BASE_MODULE_NAME}.np_backend",
            sources = [f"{BASE_SRC_PATH}/np_backend.c"],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            include_dirs = include_dirs)
    )
else:
    # raw implementation
    # NOTE: not implemented now
    extensions.append(
        Extension(
            f"{BASE_MODULE_NAME}.raw_backend",
            sources = [f"{BASE_SRC_PATH}/raw_backend.c"])
    )

# frame
extensions.append(
    Extension(
        f"{BASE_MODULE_NAME}.frame",
        sources = [f"{BASE_SRC_PATH}/frame.c"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        include_dirs=include_dirs)
)

setup(
    name="pymaro",
    version=__version__,
    description=pk_description,
    long_description=pk_long_description,
    author="Arthur Jiang",
    author_email="shujia.jiang@microsoft.com",
    url="https://github.com/microsoft/maro",
    project_urls={
        "Code": "https://github.com/microsoft/maro",
        "Issues": "https://github.com/microsoft/maro/issues",
        "Documents": "https://maro.readthedocs.io/en/latest"
    },
    license=pk_license,
    platforms=["Windows", "Linux", "macOS"],
    keywords=["citi-bike", "inventory-management", "operations-research", "reinforcement-learning", "resource-optimization", "simulator"],
    classifiers=[
        # See <https://pypi.org/classifiers/> for all classifiers
        "Programing Language :: Python",
        "Programing Language :: Python :: 3"
    ],
    python_requires=">=3.6,<3.8",
    setup_requires=[
        "numpy==1.19.1",
    ],
    install_requires=[
        # TODO: use a helper function to collect these
        "numpy==1.19.1",
        "torch==1.6.0",
        "holidays==0.10.3",
        "pyaml==20.4.0",
        "redis==3.5.3",
        "pyzmq==19.0.2",
        "requests==2.24.0",
        "psutil==5.7.2",
        "deepdiff==5.0.2",
        "azure-storage-blob==12.3.2",
        "azure-storage-common==2.1.0",
        "geopy==2.0.0",
        "pandas==0.25.3",
        "PyYAML==5.3.1"
    ],
    entry_points={
        "console_scripts": [
            "maro=maro.cli.maro:main",
        ]
    },
    packages=find_packages(exclude=["examples", "examples.*"]),
    include_package_data=True,
    package_data={
        "maro.simulator.scenarios.cim": ["topologies/*/*.yml", "meta/*.yml"],
        "maro.simulator.scenarios.citi_bike": ["topologies/*/*.yml", "meta/*.yml"],
        "maro.cli.k8s": ["lib/**/*"],
        "maro.cli.grass": ["lib/**/*"],
    },
    zip_safe=False,
    ext_modules=extensions,
)
