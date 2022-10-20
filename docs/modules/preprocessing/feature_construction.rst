Feature construction
====================

When dealing with functional data we might want to construct new features
that can be used as additional inputs to the machine learning algorithms.
The expectation is that these features make explicit characteristics that
facilitate the learning process.


Feature union
-------------

This transformer defines a way of extracting a high number of distinct
features in parallel.

.. autosummary::
   :toctree: autosummary

   skfda.preprocessing.feature_construction.FDAFeatureUnion


Per class transformer
---------------------

This method deals with the extraction of features using the information of
the target classes It applies as many transformations as classes
to every observation. Each transformation is fitted using only the training
data of a particular class.

.. autosummary::
   :toctree: autosummary

   skfda.preprocessing.feature_construction.PerClassTransformer
   

Functional features
-------------------

The following transformers can be used to extract specific functional features
for each functional datum, which can then be used for prediction.

.. autosummary::
   :toctree: autosummary
   
   skfda.preprocessing.feature_construction.LocalAveragesTransformer
   skfda.preprocessing.feature_construction.OccupationMeasureTransformer
   skfda.preprocessing.feature_construction.NumberCrossingsTransformer
   
   