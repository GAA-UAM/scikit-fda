Scaling
=======================

In the context of vector valued functional data and mixed data, scaling and 
centering becomes a important aspect to consider before applying any analysis
\ :footcite:p:`prothero++_2023_new,happ+greven_2018_multivariate,suyundykov++_2010_multivariate`.
In classical functional data analysis, centering and scaling functional data 
is not considered in the literature as a normal practice. When you have a 
set of functions, where all of them are sampled in the same domain, measure 
the same phenomenon and are measured in the same units, it is not necessary 
to center or scale the data: distances between the functions are already 
meaningful. However, when you have a vector valued functional dataset or a 
mixed dataset, where each component is measured in different units or scales,
it becomes necessary to center and scale the data before applying any analysis.

This module provides transformers that follow the scikit-learn API and apply
centering and/or scaling to functional data represented as either
:class:`~skfda.representation.grid.FDataGrid` or
:class:`~skfda.representation.basis.FDataBasis`. These scalers can be
seamlessly integrated into machine learning pipelines.

The two main classes offered are
:class:`~skfda.preprocessing.scaler.StandardScaler` and
:class:`~skfda.preprocessing.scaler.CenterScaler`.
The former computes the empirical mean and standard deviation, while the
latter allows more flexible definitions of centering and scaling
transformations.

.. note::

   All scalers are compatible with both basis and discretized representations
   of functional data.

General Scalers
---------------

These classes apply centering and scaling transformations to functional data.
The centering and scaling parameters can be user-defined (e.g., using constants
or :class:`~skfda.representation.FData` objects) or learned from the data via a
callable.

.. autosummary::
   :toctree: autosummary

   skfda.preprocessing.scaler.CenterScaler

Standardization
---------------

The :class:`~skfda.preprocessing.scaler.StandardScaler` computes the mean and
standard deviation from the training data and uses them to transform new data.
This class mimics the behavior of scikit-learn's standard scaler, but for
functional data.

It also accepts a degrees of freedom correction parameter, useful in
statistical settings that require unbiased estimators.

.. autosummary::
   :toctree: autosummary

   skfda.preprocessing.scaler.StandardScaler

Transformation Logic
--------------------

Internally, these scalers apply pointwise operations over the data.
For basis representations, the transformations are carried out using
evaluation and re-projection into the original basis. For discretized data,
direct operations on the data matrix are performed.

Custom transformations can be defined by passing functions to the ``center``
and ``scale`` parameters of :class:`~skfda.preprocessing.scaler.CenterScaler`.

References
----------

.. footbibliography::

