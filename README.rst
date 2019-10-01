.. image:: https://raw.githubusercontent.com/GAA-UAM/scikit-fda/develop/docs/logos/title_logo/title_logo.png
	:alt: scikit-fda: Functional Data Analysis in Python

scikit-fda: Functional Data Analysis in Python
===================================================

|python|_ |build-status| |docs| |Codecov|_ |PyPIBadge|_ |license|_

Functional Data Analysis, or FDA, is the field of Statistics that analyses
data that depend on a continuous parameter.

This package offers classes, methods and functions to give support to FDA
in Python. Includes a wide range of utils to work with functional data, and its
representation, exploratory analysis, or preprocessing, among other tasks
such as inference, classification, regression or clustering of functional data.
See documentation for further information on the features included in the
package.

Documentation
=============

The documentation is available at
`fda.readthedocs.io/en/stable/ <https://fda.readthedocs.io/en/stable/>`_, which
includes detailed information of the different modules, classes and methods of
the package, along with several examples showing different funcionalities.

The documentation of the latest version, corresponding with the develop
version of the package, can be found at
`fda.readthedocs.io/en/latest/ <https://fda.readthedocs.io/en/latest/>`_.

Installation
============
Currently, *scikit-fda* is available in Python 3.6 and 3.7, regardless of the
platform.
The stable version can be installed via PyPI_:

.. code::

    pip install scikit-fda

Installation from source
------------------------

It is possible to install the latest version of the package, available in the
develop branch,  by cloning this repository and doing a manual installation.

.. code::

    git clone https://github.com/GAA-UAM/scikit-fda.git
    cd scikit-fda/
    pip install -r requirements.txt    # Install dependencies
    python setup.py install

Make sure that your default Python version is currently supported, or change
the python and pip commands by specifying a version, such as ``python3.6``:

.. code::

    git clone https://github.com/GAA-UAM/scikit-fda.git
    cd scikit-fda/
    python3.6 -m pip install -r requirements.txt    # Install dependencies
    python3.6 setup.py install

Requirements
------------
*scikit-fda* depends on the following packages:

* `setuptools <https://github.com/pypa/setuptools>`_ - Python Packaging
* `cython <https://github.com/cython/cython>`_ - Python to C compiler
* `numpy <https://github.com/numpy/numpy>`_ - The fundamental package for scientific computing with Python
* `pandas <https://github.com/pandas-dev/pandas>`_ - Powerful Python data analysis toolkit
* `scipy <https://github.com/scipy/scipy>`_ - Scientific computation in Python
* `scikit-learn <https://github.com/scikit-learn/scikit-learn>`_ - Machine learning in Python
* `matplotlib <https://github.com/matplotlib/matplotlib>`_ - Plotting with Python
* `mpldatacursor <https://github.com/joferkington/mpldatacursor/>`_ - Interactive data cursors for matplotlib
* `rdata <https://github.com/vnmabus/rdata>`_ - Reader of R datasets in .rda format in Python
* `scikit-datasets <https://github.com/daviddiazvico/scikit-datasets>`_ - Scikit-learn compatible datasets

The dependencies are automatically installed.

Contributions
=============
All contributions are welcome. You can help this project grow in multiple ways,
from creating an issue, reporting an improvement or a bug, to doing a
repository fork and creating a pull request to the development branch.

The people involved at some point in the development of the package can be
found in the `contributors
file <https://github.com/GAA-UAM/scikit-fda/blob/develop/THANKS.txt>`_.

Citation
========
If you find this project useful, please cite:

.. todo:: Include citation to scikit-fda paper.

License
=======

The package is licensed under the BSD 3-Clause License. A copy of the
license_ can be found along with the code.

.. _examples: https://fda.readthedocs.io/en/latest/auto_examples/index.html
.. _PyPI: https://pypi.org/project/scikit-fda/

.. |python| image:: https://img.shields.io/pypi/pyversions/scikit-fda.svg
.. _python: https://badge.fury.io/py/scikit-fda

.. |build-status| image:: https://travis-ci.org/GAA-UAM/scikit-fda.svg?branch=develop
    :alt: build status
    :scale: 100%
    :target: https://travis-ci.org/GAA-UAM/scikit-fda

.. |docs| image:: https://readthedocs.org/projects/fda/badge/?version=latest
    :alt: Documentation Status
    :scale: 100%
    :target: http://fda.readthedocs.io/en/latest/?badge=latest

.. |Codecov| image:: https://codecov.io/gh/GAA-UAM/scikit-fda/branch/develop/graph/badge.svg
.. _Codecov: https://codecov.io/github/GAA-UAM/scikit-fda?branch=develop

.. |PyPIBadge| image:: https://badge.fury.io/py/scikit-fda.svg
.. _PyPIBadge: https://badge.fury.io/py/scikit-fda

.. |license| image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
.. _license: https://github.com/GAA-UAM/scikit-fda/blob/master/LICENSE.txt
