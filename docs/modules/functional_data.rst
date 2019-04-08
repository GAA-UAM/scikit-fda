Functional Data
===============

Before beginning to use the functionalities of the package, it is necessary to
represent the data in functional form, using one of the following classes,
which allow the visualization, evaluation and treatment of the data in a simple
way, using the advantages of the object-oriented programming.

Discrete representation
-----------------------

A functional datum may be treated using a non-parametric representation,
storing the values of the functions in a finite grid of points. The FDataGrid
class supports multivariate functions using this approach. In the
`discretized function representation example
<../auto_examples/plot_discrete_representation.html>`_ it is shown the creation
and basic visualisation of a FDataGrid.


.. autosummary::
   :toctree: autosummary

   fda.grid.FDataGrid


Functional data grids may be evaluated using interpolation, as it  is shown in
the `interpolation example <../auto_examples/plot_interpolation.html>`_. The
following class allows interpolation with different splines.

.. autosummary::
   :toctree: autosummary

   fda.grid.GridSplineInterpolator


Basis representation
--------------------

The package supports a parametric representation using a linear combination
of elements of a basis function system.

.. autosummary::
   :toctree: autosummary

   fda.FDataBasis


The following classes are used to define different basis systems.

.. autosummary::
   :toctree: autosummary

   fda.basis.BSpline
   fda.basis.Fourier
   fda.basis.Monomial

Generic representation
----------------------

Functional objects of the package are instances of FData, which
contains the common attributes and methods used in all representations. This
is an abstract class and cannot be instantiated directly, because it does not
specify the representation of the data. Many of the package's functionalities
receive an element of this class as an argument. 

.. autosummary::
   :toctree: autosummary

   fda.functional_data.FData
