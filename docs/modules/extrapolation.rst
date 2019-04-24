Extrapolation
=============

This module contains the extrapolators used to evaluate points outside the
domain range of :class:`FDataBasis` or :class:`FDataGrid`. See
`Extrapolation Example
<../auto_examples/plot_extrapolation.html>`_ for detailed explanation.

Extrapolation Methods
---------------------

The following classes are used to define common methods of extrapolation.

.. autosummary::
   :toctree: autosummary

   skfda.extrapolation.BoundaryExtrapolation
   skfda.extrapolation.ExceptionExtrapolation
   skfda.extrapolation.FillExtrapolation
   skfda.extrapolation.PeriodicExtrapolation

Custom Extrapolation
--------------------

Custom extrapolators could be done subclassing :class:`Extrapolator
<skfda.extrapolation.Extrapolator>` or with a compatible callable.

.. autosummary::
   :toctree: autosummary

   skfda.extrapolation.Extrapolator
