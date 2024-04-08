Extrapolation
=============

This module contains the extrapolators used to evaluate points outside the
:term:`domain` range of :class:`FDataBasis` or :class:`FDataGrid`. See
`Extrapolation Example
<../auto_examples/plot_extrapolation.html>`_ for detailed explanation.

Extrapolation Methods
---------------------

The following classes are used to define common methods of extrapolation.

.. autosummary::
   :toctree: autosummary

   skfda.representation.extrapolation.BoundaryExtrapolation
   skfda.representation.extrapolation.ExceptionExtrapolation
   skfda.representation.extrapolation.FillExtrapolation
   skfda.representation.extrapolation.PeriodicExtrapolation
