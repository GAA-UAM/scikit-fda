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

The following functions can be used to create new features from functional data.

.. autosummary::
   :toctree: autosummary

   skfda.exploratory.stats.modified_epigraph_index
   skfda.preprocessing.feature_construction.local_averages
   skfda.preprocessing.feature_construction.occupation_measure
   skfda.preprocessing.feature_construction.number_crossings

Some of them are also available as transformers that can be directly used in a
pipeline: 

.. autosummary::
   :toctree: autosummary
   
   skfda.preprocessing.feature_construction.LocalAveragesTransformer
   skfda.preprocessing.feature_construction.OccupationMeasureTransformer
   skfda.preprocessing.feature_construction.NumberCrossingsTransformer
   