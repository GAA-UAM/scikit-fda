Conversion between representations
==================================

This module contains classes (converters) for converting between different
representations. Currently only the conversion between :class:`FDataIrregular`
and :class:`FDataBasis` has been implemented via converters.

:class:`FDataIrregular` to :class:`FDataBasis`
----------------------------------------------

The following module contains a detailed description of one of the types
of converters.

.. autosummary::
   :toctree: autosummary

   skfda.representation.conversion._mixed_effects


The following classes are used for converting irregular functional
data to basis representation using the mixed effects model.

.. autosummary::
   :toctree: autosummary

   skfda.representation.conversion.EMMixedEffectsConverter
   skfda.representation.conversion.MinimizeMixedEffectsConverter
   skfda.representation.conversion.MixedEffectsConverter

