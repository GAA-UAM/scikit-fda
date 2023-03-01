
.. image:: logos/title_logo/title_logo.png
	:alt: scikit-fda: Functional Data Analysis in Python

Welcome to scikit-fda's documentation!
======================================

This package offers classes, methods and functions to give support to
Functional Data Analysis in Python. Includes a wide range of utils to work with
functional data, and its representation, exploratory analysis, or
preprocessing, among other tasks such as inference, classification, regression
or clustering of functional data.

In the `project page <https://github.com/GAA-UAM/scikit-fda>`_ hosted by
Github you can find more information related to the development of the package.

.. toctree::
   :caption: Using scikit-fda
   :hidden:
	
   auto_tutorial/index

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :hidden:

   auto_examples/index

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :hidden:
   :caption: More documentation

   apilist
   glossary

An exhaustive list of all the contents of the package can be found in the
:ref:`genindex`.

Installation
------------

Currently, *scikit-fda* is available in Python versions above 3.8, regardless of the
platform.
The stable version can be installed via
`PyPI <https://pypi.org/project/scikit-fda/>`_:

.. code-block:: bash

   pip install scikit-fda

It is also available from conda-forge:

.. code-block:: bash

    conda install -c conda-forge scikit-fda

It is possible to install the latest version of the package, available in
the develop branch, by cloning this repository and doing a manual installation.

.. code-block:: bash

   git clone https://github.com/GAA-UAM/scikit-fda.git
   pip install ./scikit-fda


In this type of installation make sure that your default Python version is
currently supported, or change the python and pip commands by specifying a
version, such as python3.6.

How do I start?
---------------

If you want a quick overview of the package, we recommend you to try the
new :doc:`tutorial <auto_tutorial/index>`. For articles about specific
topics, feel free to explore the :doc:`examples <auto_examples/index>`. Want
to check the documentation of a particular class or function? Try searching
for it in the :doc:`API list <apilist>`.

Contributions
-------------

All contributions are welcome. You can help this project grow in multiple ways,
from creating an issue, reporting an improvement or a bug, to doing a
repository fork and creating a pull request to the development branch.
The people involved at some point in the development of the package can be
found in the `contributors file
<https://github.com/GAA-UAM/scikit-fda/blob/develop/THANKS.txt>`_.

.. Citation
   --------
   If you find this project useful, please cite:

   .. todo:: Include citation to scikit-fda paper.

License
-------

The package is licensed under the BSD 3-Clause License. A copy of the
`license <https://github.com/GAA-UAM/scikit-fda/blob/develop/LICENSE.txt>`_
can be found along with the code or in the project page.
