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
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import Bunch

import skfda
from skfda.datasets._real_datasets import fetch_bone_density
from skfda.preprocessing.dim_reduction import PACE
from skfda.representation import FDataIrregular

# %%
# In this example we are going to use functional principal component analysis
# through conditional expectation to explore datasets and obtain conclusions
# about said dataset using this technique.
cd4_bunch: Bunch = fetch_bone_density()
cd4: FDataIrregular = cd4_bunch.data
assert isinstance(cd4, FDataIrregular), "Expected an FDataIrregular object"

cd4[:20].plot()
plt.show()

# %%
plt.figure()
for t, v in zip(cd4.points, cd4.values, strict=True):
    t_i = np.asarray(t)
    v_i = np.asarray(v)
    plt.scatter(t_i, v_i, alpha=0.7, s=10, color="black")

plt.xlabel("months since seroconversion")
plt.ylabel("CD4 cell count")
plt.title("All observed CD4 values across subjects")
plt.tight_layout()
plt.show()
# %%
pace = PACE(n_components=0.95)
pace.fit(cd4)

fpc_scores = pace.transform(cd4)

# %%
