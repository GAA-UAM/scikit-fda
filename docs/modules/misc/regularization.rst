Regularization
==============

This module contains several regularization techniques that can be applied
in several situations, such as regression, PCA or basis smoothing.

These regularization methods are useful to obtain simple solutions and to
introduce known hypothesis to the model, such as periodicity or smoothness,
reducing the effects caused by noise in the observations.

In functional data analysis is also common to have ill posed problems, because
of the infinite nature of the data and the finite sample size. The application
of regularization techniques in these kind of problems is then necessary to
obtain reasonable solutions.

When dealing with multivariate data, a common choice for the regularization
is to penalize the squared Euclidean norm, or :math:`L_2` norm, of the vectors
in order to obtain simpler solutions. This can be done in scikit-fda for
both multivariate and functional data using the :class:`L2Regularization`
class. A more flexible generalization of this approach is to penalize the
squared :math:`L_2` norm after a particular linear operator is
applied. This for example allows to penalize the second derivative of a curve,
which is a measure of its curvature, because the differential operator
is linear. As arbitrary Python callables can be used as operators (provided
that they correspond to a linear transformation), it is possible to penalize
the evaluation at a point, the difference between points or other arbitrary
linear operations.

.. autosummary::
   :toctree: autosummary

   skfda.misc.regularization.L2Regularization