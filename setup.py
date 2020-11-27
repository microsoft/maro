# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import io
import os
import numpy

from setuptools import Extension, find_packages, setup

from maro import __version__

# Set environment variable to skip deployment process of MARO
os.environ["SKIP_DEPLOYMENT"] = "TRUE"


def gen_backend_extensions(extensions: list):
    # root path to backend
    BASE_SRC_PATH = "./maro/backends"
    # backend module name
    BASE_MODULE_NAME = "maro.backends"

    # include dirs for frame and its backend
    include_dirs = []

    # backend base extensions
    extensions.append(
        Extension(
            f"{BASE_MODULE_NAME}.backend",
            sources=[f"{BASE_SRC_PATH}/backend.cpp"],
            extra_compile_args=['-std=c++11'])
    )

    include_dirs.append(numpy.get_include())

    extensions.append(
        Extension(
            f"{BASE_MODULE_NAME}.np_backend",
            sources=[f"{BASE_SRC_PATH}/np_backend.cpp"],
            include_dirs=include_dirs,
            extra_compile_args=['-std=c++11'])
    )

    # raw implementation
    # NOTE: not implemented now
    extensions.append(
        Extension(
            f"{BASE_MODULE_NAME}.raw_backend",
            sources=[f"{BASE_SRC_PATH}/raw_backend.cpp"],
            include_dirs=include_dirs,
            extra_compile_args=['-std=c++11'])
    )

    # frame
    extensions.append(
        Extension(
            f"{BASE_MODULE_NAME}.frame",
            sources=[f"{BASE_SRC_PATH}/frame.cpp"],
            include_dirs=include_dirs,
            extra_compile_args=['-std=c++11'])
    )

def gen_datalib_extension(extensions: list):
    BASE_SRC_PATH = "./maro/data_lib/binary"

    BASE_MODULE_NAME = "maro.data_lib.binary"

    extensions.append(
        Extension(
            f"{BASE_MODULE_NAME}.binaryreader",
            sources=[f"{BASE_SRC_PATH}/binaryreader.cpp"],
            extra_compile_args=['-std=c++11']
        )
    )

    extensions.append(
        Extension(
            f"{BASE_MODULE_NAME}.binaryconverter",
            sources=[f"{BASE_SRC_PATH}/binaryconverter.cpp"],
            extra_compile_args=['-std=c++11']
        )
    )


# extensions to be compiled
extensions = []

gen_backend_extensions(extensions)
gen_datalib_extension(extensions)


readme = io.open("./maro/README.rst", encoding="utf-8").read()

setup(
    name="pymaro",
    version=__version__,
    description="MARO Python Package",
    long_description=readme,
    long_description_content_type="text/x-rst",
    author="Arthur Jiang",
    author_email="shujia.jiang@microsoft.com",
    url="https://github.com/microsoft/maro",
    project_urls={
        "Code": "https://github.com/microsoft/maro",
        "Issues": "https://github.com/microsoft/maro/issues",
        "Documents": "https://maro.readthedocs.io/en/latest"
    },
    license="MIT License",
    platforms=["Windows", "Linux", "macOS"],
    keywords=[
        "citi-bike",
        "inventory-management",
        "operations-research",
        "reinforcement-learning",
        "resource-optimization",
        "simulator"],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"],
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
        "azure-storage-blob==12.6.0",
        "azure-storage-common==2.1.0",
        "geopy==2.0.0",
        "pandas==0.25.3",
        "PyYAML==5.3.1",
        "paramiko==2.7.2"
    ],
    entry_points={
        "console_scripts": [
            "maro=maro.cli.maro:main",
        ]
    },
    packages=find_packages(exclude=["examples", "examples.*"]),
    include_package_data=True,
    package_data={
        "maro.simulator.scenarios.cim": ["topologies/*/*.yml", "meta/*.toml"],
        "maro.simulator.scenarios.citi_bike": ["topologies/*/*.yml", "meta/*.yml", "meta/*.toml"],
        "maro.cli.k8s": ["lib/*", "lib/*/*", "lib/*/*/*"],
        "maro.cli.grass": ["lib/*", "lib/*/*", "lib/*/*/*"],
    },
    zip_safe=False,
    ext_modules=extensions,
)
