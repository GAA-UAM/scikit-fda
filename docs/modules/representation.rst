Representation of functional Data
=================================

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

   skfda.representation.grid.FDataGrid


Functional data grids may be evaluated using interpolation, as it  is shown in
the `interpolation example <../auto_examples/plot_interpolation.html>`_. The
following class allows interpolation with different splines.

.. autosummary::
   :toctree: autosummary

   skfda.representation.interpolation.SplineInterpolation


Basis representation
--------------------

The package supports a parametric representation using a linear combination
of elements of a basis function system.

.. autosummary::
   :toctree: autosummary

   skfda.representation.basis.FDataBasis


The following classes are used to define different basis for
:math:`\mathbb{R} \to \mathbb{R}` functions.

.. autosummary::
   :toctree: autosummary

   skfda.representation.basis.BSplineBasis
   skfda.representation.basis.FourierBasis
   skfda.representation.basis.MonomialBasis
   skfda.representation.basis.ConstantBasis
   skfda.representation.basis.CustomBasis
   
The following classes, allow the construction of a basis for
:math:`\mathbb{R}^n \to \mathbb{R}` functions.

.. autosummary::
   :toctree: autosummary

   skfda.representation.basis.TensorBasis
   skfda.representation.basis.FiniteElementBasis

The following class, allows the construction of a basis for
:math:`\mathbb{R}^n \to \mathbb{R}^m` functions from
several :math:`\mathbb{R}^n \to \mathbb{R}` bases.

.. autosummary::
   :toctree: autosummary

   skfda.representation.basis.VectorValuedBasis
   
All the aforementioned basis inherit the basics from an
abstract base class :class:`Basis`. Users can create their own
basis subclassing this class and implementing the required
methods.

.. autosummary::
   :toctree: autosummary

   skfda.representation.basis.Basis


Irregular representation
------------------------

In practice, most functional datasets do not contain functions evaluated
uniformly over a fixed grid. In other words, it is paramount to be able
to represent irregular functional data.

While the FDataGrid class could support these kind of datasets, it is
inefficient to store a complete grid with low data density. Furthermore,
there are specific methods that can be applied to irregular data in order
to obtain, among other things, a better convesion to basis representation.

The FDataIrregular class provides the functionality which suits these purposes.


.. autosummary::
   :toctree: autosummary

   skfda.representation.irregular.FDataIrregular


Generic representation
----------------------

Functional objects of the package are instances of FData, which
contains the common attributes and methods used in all representations. This
is an abstract class and cannot be instantiated directly, because it does not
specify the representation of the data. Many of the package's functionalities
receive an element of this class as an argument.

.. autosummary::
   :toctree: autosummary

   skfda.representation.FData

Extrapolation
-------------
All representations of functional data allow evaluation outside of the original
interval using extrapolation methods.

.. toctree::
   :maxdepth: 4

   representation/extrapolation

Deprecated Classes
----------------------

.. autosummary::
   :toctree: autosummary

   skfda.representation.basis.BSpline
   skfda.representation.basis.Fourier
   skfda.representation.basis.Monomial
   skfda.representation.basis.Constant
   skfda.representation.basis.Tensor
   skfda.representation.basis.FiniteElement
   skfda.representation.basis.VectorValued

