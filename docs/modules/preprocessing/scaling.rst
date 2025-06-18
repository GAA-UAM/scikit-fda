Scaling
=======================

In the context of vector valued functional data and mixed data, scaling and 
centring becomes a important aspect to consider before applying any analysis.
In classical functional data analysis, centring and scaling functional data 
is not considered in the literature as a normal practice. When you have a 
set of functions, where all of them are sampled in the same domain, measure 
the same phenomenon and are measured in the same units, it is not necessary 
to centre or scale the data, distances between the functions are already 
meaningful. However, when we have a vector valued functional dataset or a 
mixed dataset, where each component is measured in different units or scales,
it becomes necessary to centre and scale the data before applying any analysis.

This module provides transformers that follow the scikit-learn API and apply
centering and/or scaling to functional data represented as either
``FDataGrid`` or ``FDataBasis``. These scalers can be seamlessly integrated
into machine learning pipelines.

The two main classes offered are ``StandardScaler`` and ``CenterScaler``.
The former computes the empirical mean and standard deviation, while the
latter allows more flexible definitions of centering and scaling transformations.

.. note::

   All scalers are compatible with both basis and discretized representations
   of functional data.

General Scalers
---------------

These classes apply centering and scaling transformations to functional data.
The centering and scaling parameters can be user-defined (e.g., using constants
or ``FData`` objects) or learned from the data via a callable.

.. autosummary::
   :toctree: autosummary

   skfda.preprocessing.scaler.CenterScaler

Standardization
---------------

The ``StandardScaler`` computes the mean and standard deviation from the
training data and uses them to transform new data. This class mimics the
behavior of scikit-learn's standard scaler, but for functional data.

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
and ``scale`` parameters of ``CenterScaler``.

References
----------

* J. Prothero, J. Hannig, and J. Marron. *New perspectives on centering*. The New England Journal of Statistics in Data Science, **1**(2), 216–236, 2023.

* C. Happ and S. Greven. *Multivariate Functional Principal Component Analysis for Data Observed on Different (Dimensional) Domains*. Journal of the American Statistical Association, **113**(522), 649–659, 2018.

* S. Suyundykov, S. Puechmorel, and L. Ferré. *Multivariate functional data clusterization by PCA in Sobolev space using wavelets*. In *42èmes Journées de Statistique*, 2010.

