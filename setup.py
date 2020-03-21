# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro import __version__
from setuptools import find_packages
from distutils.core import setup, Extension
from Cython.Distutils import build_ext

import numpy

# include c file if needed
frame_ext = Extension("maro.simulator.frame",
          sources=["maro/simulator/frame/frame.pyx"],
          include_dirs=[numpy.get_include()])

frame_ext.cython_directives = {"embedsignature": True}

setup(
    name="maro",
    version=__version__,
    description="",
    long_description="",
    long_description_content_type="text/markdown",
    author="",
    author_email="",
    url="",
    project_urls={
        "Code": "/path/to/code/repo",
        "Issues": "/path/to/issues",
        "Documents": "/path/to/documents"
    },
    license="",
    platforms=[],
    cmdclass={
        "build_ext": build_ext
    },
    keywords=[],
    classifiers=[
        # See <https://pypi.org/classifiers/> for all classifiers
        "Programing Language :: Python",
        "Programing Language :: Python :: 3"
    ],
    python_requires=">=3",
    setup_requires=[
        'numpy',
        'Cython',
    ],
    install_requires=[
        # TODO: use a helper function to collect these
        'numpy',
        'pyaml',
        'redis',
        'pyzmq',
        'influxdb',
        'requests',
        'psutil',
    ],
    entry_points={
        "console_scripts": [
            'maro=maro.cli.maro:main',
        ]
    },
    packages=find_packages(),
    package_data={
        # include our configs here
        "maro.simulator.scenarios.ecr": ["topologies/*/*.yml"],

        # TODO: more data from other modules
    },
    data_files=[
        ('maro_dashboard', ['maro/utils/dashboard/resource.tar.gz'])
    ],
    ext_modules=[
        frame_ext,
    ],
    zip_safe=False
)
