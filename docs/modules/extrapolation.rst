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

   skfda.BoundaryExtrapolation
   skfda.ExceptionExtrapolation
   skfda.FillExtrapolation
   skfda.PeriodicExtrapolation

Custom Extrapolation
--------------------

Custom extrapolators could be done subclassing :class:`EvaluatorConstructor
<skfda.EvaluatorConstructor>`.

.. autosummary::
   :toctree: autosummary

   skfda.EvaluatorConstructor
   skfda.Evaluator
