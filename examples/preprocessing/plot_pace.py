"""
Functional Principal Component Analysis through Conditional Expectation
=======================================================================

Explores an alternative way to do functional principal component analysis for
irregularly sampled data.
"""

# Author: Alejandro Arias Gomez
# License: MIT

# sphinx_gallery_thumbnail_number = -1

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import Bunch

import skfda
from skfda.datasets._real_datasets import fetch_cd4
from skfda.preprocessing.dim_reduction import PACE
from skfda.representation import FDataIrregular

# %%
# In this example we are going to use functional principal component analysis
# through conditional expectation to explore datasets and obtain conclusions
# about said dataset using this technique.
#
# The PACE algorithm is an alternative to FPCA that is specifically designed
# for irregularly sampled data. It uses local linear smoothing to estimate
# the model components of the data, and then performs the conditional
# expectation step to obtain the principal components. These components are
# the directions that capture the main modes of variation across the function.
# PACE shares the same objectives as FPCA, it is a dimensionality
# reduction method for functional data that aims to reduce the complexity of
# studying observations by expressing the data in terms of a basis of K
# components that explain most of the variation in the data.
#
# The PACE algorithm was introduced in :footcite:ts`yao+muller+wang_2005_pace`,
# and in this example we will use the implementation of the algorithm that
# follows the same steps as the original algorithm.
#
# We will analyse one of the datasets that is used in the original paper, the
# CD4 dataset. This dataset contains the CD4 cell counts of 366 HIV patients
# measured in between months -18 and 42 since seroconversion. To better
# understand the data, we will plot the first 20 subjects.
cd4_bunch: Bunch = fetch_cd4()
cd4: FDataIrregular = cd4_bunch.data
assert isinstance(cd4, FDataIrregular), "Expected an FDataIrregular object"

cd4[:20].plot()
plt.show()
# %%
# Continuing with the analysis of the dataset, we will plot the total data
# across all subjects, where we can further identify the sparsity of the data.
# The data is spread across all the domain, with more frequent measurements
# every trimester, especially the first one before and after seroconversion.
plt.figure(figsize=(10, 6))
for i, (t, _) in enumerate(zip(cd4.points, cd4.values, strict=True)):
    plt.scatter(t, [i] * len(t), alpha=0.7, s=10, color="black")
plt.xlabel("months since seroconversion")
plt.ylabel("CD4 cell count")
plt.title("Total observed points in the CD4 dataset")
plt.tight_layout()
plt.show()

# %%
# We can now apply the PACE method to the dataset to obtain the principal
# components of the data.
pace = PACE(n_components=3)
pace.fit(cd4)

pace.components_.plot()

# %%
