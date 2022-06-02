# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# isort: skip_file

import io
import os
import sys
import numpy

# NOTE: DO NOT change the import order, as sometimes there is a conflict between setuptools and distutils,
# it will cause following error:
# error: each element of 'ext_modules' option must be an Extension instance or 2-tuple
from setuptools import find_packages
from distutils.core import setup
from distutils.extension import Extension

from maro import __version__

compile_flag = "-std=c++11"

if sys.platform == "win32":
    compile_flag = "/std:c++14"

# Set environment variable to skip deployment process of MARO
os.environ["SKIP_DEPLOYMENT"] = "TRUE"


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

# include dirs for frame and its backend
include_dirs = []

# backend base extensions
extensions.append(
    Extension(
        f"{BASE_MODULE_NAME}.backend",
        sources=[f"{BASE_SRC_PATH}/backend.cpp"],
        extra_compile_args=[compile_flag],
    ),
)


include_dirs.append(numpy.get_include())

extensions.append(
    Extension(
        f"{BASE_MODULE_NAME}.np_backend",
        sources=[f"{BASE_SRC_PATH}/np_backend.cpp"],
        include_dirs=include_dirs,
        extra_compile_args=[compile_flag],
    ),
)

# raw implementation
# NOTE: not implemented now
extensions.append(
    Extension(
        f"{BASE_MODULE_NAME}.raw_backend",
        sources=[f"{BASE_SRC_PATH}/raw_backend.cpp"],
        include_dirs=include_dirs,
        extra_compile_args=[compile_flag],
    ),
)

# frame
extensions.append(
    Extension(
        f"{BASE_MODULE_NAME}.frame",
        sources=[f"{BASE_SRC_PATH}/frame.cpp"],
        include_dirs=include_dirs,
        extra_compile_args=[compile_flag],
    ),
)

specific_requires = []
if sys.version.startswith("3.6"):
    specific_requires.append("dataclasses>=0.5")

readme = io.open("./maro/README.rst", encoding="utf-8").read()

setup(
    name="pymaro",
    version=__version__,
    description="MARO Python Package",
    long_description=readme,
    long_description_content_type="text/x-rst",
    author="MARO Team",
    author_email="maro-team@microsoft.com",
    url="https://github.com/microsoft/maro",
    project_urls={
        "Code": "https://github.com/microsoft/maro",
        "Issues": "https://github.com/microsoft/maro/issues",
        "Documents": "https://maro.readthedocs.io/en/latest",
    },
    license="MIT License",
    platforms=["Windows", "Linux", "macOS"],
    keywords=[
        "citi-bike",
        "inventory-management",
        "operations-research",
        "reinforcement-learning",
        "resource-optimization",
        "simulator",
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    setup_requires=[
        "numpy<1.20.0",
    ],
    install_requires=[
        # TODO: use a helper function to collect these
        "numpy<1.20.0",
        "scipy<=1.7.0",
        "torch<1.8.0",
        "holidays>=0.10.3",
        "pyaml>=20.4.0",
        "redis>=3.5.3",
        "pyzmq<22.1.0",
        "requests<=2.26.0",
        "psutil<5.9.0",
        "deepdiff>=5.2.2",
        "azure-storage-blob<12.9.0",
        "azure-storage-common",
        "geopy>=2.0.0",
        "pandas<1.2",
        "PyYAML<5.5.0",
        "paramiko>=2.7.2",
        "kubernetes>=12.0.1",
        "prompt_toolkit<3.1.0",
        "stringcase>=1.2.0",
    ]
    + specific_requires,
    entry_points={
        "console_scripts": [
            "maro=maro.cli.maro:main",
        ],
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    include_package_data=True,
    package_data={
        "maro.simulator.scenarios.cim": ["topologies/*/*.yml", "meta/*.yml"],
        "maro.simulator.scenarios.citi_bike": ["topologies/*/*.yml", "meta/*.yml"],
        "maro.simulator.scenarios.vm_scheduling": ["topologies/*/*.yml", "meta/*.yml"],
        "maro.cli.k8s": ["lib/*", "lib/*/*", "lib/*/*/*", "lib/*/*/*/*"],
        "maro.cli.grass": ["lib/*", "lib/*/*", "lib/*/*/*", "lib/*/*/*/*", "lib/*/*/*/*/*"],
        "maro.cli.project_generator/templates": ["*.jinja"],
        "maro.cli.utils": ["web_terminal/*"],
    },
    zip_safe=False,
    ext_modules=extensions,
)
