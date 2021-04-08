"""
Getting the data
================

How to get data to use in scikit-fda.

"""

# Author: Carlos Ramos Carreño
# License: MIT

##############################################################################
# The FDataGrid class
# -------------------
#
# In order to use scikit-fda, first we need functional data to analyze.
# A common case is to have each functional observation measured at the same
# points.
# This kind of functional data is easily representable in scikit-fda using
# the :class:`~skfda.representation.grid.FDataGrid` class.
#
# The :class:`~skfda.representation.grid.FDataGrid` has two important
# attributes: ``data_matrix`` and ``grid_points``. The attribute
# ``grid_points`` is a tuple with the same length as the number of domain
# dimensions (that is, one for curves, two for surfaces...). Each of its
# elements is a 1D numpy :class:`~numpy.ndarray` containing the measurement
# points for that particular dimension. The attribute ``data_matrix`` is a
# numpy :class:`~numpy.ndarray` containing the measured values of the
# functions in the grid spanned by the grid points. For functions
# :math:`\{f_i: \mathbb{R}^p \to \mathbb{R}^q\}_{i=1}^N` this is a tensor
# with dimensions :math:`N \times M_1 \times \ldots \times M_p \times q`,
# where :math:`M_i` is the number of measurement points for the domain
# dimension :math:`i`.

##############################################################################
# In order to create a :class:`~skfda.representation.grid.FDataGrid`, these
# attributes may be provided. The attributes are converted to
# :class:`~numpy.ndarray` when necessary.

##############################################################################
# .. note::
#
#     The grid points can be omitted,
#     and in that case their number is inferred from the dimensions of
#     ``data_matrix`` and they are automatically assigned as equispaced points
#     in the unitary cube in the domain set.
#
#     In the common case of functions with domain dimension of 1, the list of
#     grid points can be passed directly as ``grid_points``.
#
#     If the codomain dimension is 1, the last dimension of ``data_matrix``
#     can be dropped.

##############################################################################
# The following example shows the creation of a
# :class:`~skfda.representation.grid.FDataGrid` with two functions (curves)
# :math:`\{f_i: \mathbb{R} \to \mathbb{R}\}, i=1,2` measured at the same
# (non-equispaced) points.

import skfda

grid_points = [0, 0.2, 0.5, 0.9, 1]  # Grid points of the curves
data_matrix = [
    [0, 0.2, 0.5, 0.9, 1],     # First observation
    [0, 0.04, 0.25, 0.81, 1],  # Second observation
]

fd = skfda.FDataGrid(
    data_matrix=data_matrix,
    grid_points=grid_points,
)

fd.plot()

##############################################################################
# Advanced example
# ^^^^^^^^^^^^^^^^
#
# In order to better understand the FDataGrid structure, you can consider the
# following example, in which a :class:`~skfda.representation.grid.FDataGrid`
# object is created, containing just one function (vector-valued surface)
# :math:`\{f_i: \mathbb{R}^2 \to \mathbb{R}^4`.


grid_points_surface = [
    [0.2, 0.5, 0.7],      # Measurement points in first domain dimension
    [0, 1.5],             # Measurement points in second domain dimension
]

data_matrix_surface = [
    # First observation
    [
        # 0.2
        [
            # Value at (0.2, 0)
            [1, 2, 3, 4],
            # Value at (0.2, 1.5)
            [0, 1, -1.3, 2],
        ],
        # 0.5
        [
            # Value at (0.5, 0)
            [-2, 0, 5.5, 7],
            # Value at (0.5, 1.5)
            [2, 1.1, -1, -2],
        ],
        # 0.7
        [
            # Value at (0.7, 0)
            [0, 0, 1, 1],
            # Value at (0.7, 1.5)
            [-3, 5, -0.5, -2],
        ],
    ],
    # This example has only one observation. Next observations would be
    # added here.
]

fd = skfda.FDataGrid(
    data_matrix=data_matrix_surface,
    grid_points=grid_points_surface,
)

fd.plot()

##############################################################################
# Importing data
# --------------
#
# Usually one does not construct manually the functions, but instead uses
# measurements already formatted in a common format, such as comma-separated
# values (CSV), attribute-relation file format (ARFF) or Matlab and R formats.
#
# If your data is in one of these formats, you can import it into a numpy
# array using the IO functions available in
# `Numpy <https://numpy.org/devdocs/reference/routines.io.html>`_ (for simple
# text-based or binary formats, such as CSV) or in
# `Scipy <https://docs.scipy.org/doc/scipy/reference/io.html>`_ (for Matlab,
# Fortran or ARFF files). For importing data in the R format one can also
# use the package `RData <https://rdata.readthedocs.io>`_ with is already a
# dependency of scikit-fda, as it is used to load the example datasets.

##############################################################################
# Common datasets
# ---------------
#
# scikit-fda can download and import for you several of the most popular
# datasets in the :term:`FDA` literature, such as the Berkeley Growth
# dataset (function :func:`~skfda.datasets.fetch_growth`) or the Canadian
# Weather dataset (function :func:`~skfda.datasets.fetch_weather`).

X, y = skfda.datasets.fetch_growth(return_X_y=True)

X.plot(group=y)

##############################################################################
# Datasets from CRAN
# ^^^^^^^^^^^^^^^^^^
#
# If you want to work with a dataset for which no fetching function exist, and
# you know that is available inside a R package in the CRAN repository, you
# can try using the function :func:`~skfda.datasets.fetch_cran`. This function
# will load the package, fetch the dataset and convert it to Python objects
# using the packages
# `scikit-datasets <https://github.com/daviddiazvico/scikit-datasets>`_ and
# `RData <https://rdata.readthedocs.io>`_. As datasets in CRAN follow no
# particular structure, you will need to know how it is structured internally
# in order to use it properly.

##############################################################################
# .. note::
#
#     Functional data objects from some packages, such as
#     `fda.usc <https://cran.r-project.org/web/packages/fda.usc/index.html>`_
#     are automatically recognized as such and converted to
#     :class:`~skfda.representation.grid.FDataGrid` instances. This
#     behaviour can be disabled or customized to work with more packages.

data = skfda.datasets.fetch_cran("poblenou", "fda.usc")
data["poblenou"]["nox"].plot()

##############################################################################
# In order to know all the available functionalities to load existing and
# synthetic datasets it is recommended to look at the documentation of the
# :doc:`datasets </modules/datasets>` module.
