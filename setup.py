# encoding: utf-8

"""
Functional Data Analysis Python package.

Functional Data Analysis, or FDA, is the field of Statistics that analyses
data that depend on a continuous parameter.

This package offers classes, methods and functions to give support to FDA
in Python. Includes a wide range of utils to work with functional data, and its
representation, exploratory analysis, or preprocessing, among other tasks
such as inference, classification, regression or clustering of functional data.
See documentation or visit the
`github page <https://github.com/GAA-UAM/scikit-fda>`_ of the project for
further information on the features included in the package.

The documentation is available at
`fda.readthedocs.io/en/stable/ <https://fda.readthedocs.io/en/stable/>`_, which
includes detailed information of the different modules, classes and methods of
the package, along with several examples showing different funcionalities.
"""

import os

from setuptools import find_packages, setup

with open(
    os.path.join(os.path.dirname(__file__), "VERSION"),
    "r",
) as version_file:
    version = version_file.read().strip()

setup(
    name="scikit-fda",
    version=version,
    include_package_data=True,
    packages=find_packages(),
)
