"""Maximum depth classification.

Shows the usage of maximum depth classifier.
"""

# Author: Pedro Martín Rodríguez-Ponga Eyriès
# License: MIT

import skfda
from skfda.ml.classification import MaximumDepthClassifier
from skfda.exploratory.depth import *

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

############################################################################
# In this example we are going to show the usage of the maximum depth
# classifier.
#
# Firstly, we are going to fetch a functional dataset, such as the Berkeley
# Growth Study. This dataset contains the height of several boys and girls
# measured until the 18 years of age.
# We will try to predict the sex by using its growth curves.
#
# The following figure shows the growth curves grouped by sex.
#
# Loads dataset
dataset = skfda.datasets.fetch_growth()
fd = dataset['data']
y = dataset['target']

# Plot samples grouped by sex
fd.plot(group=y)

############################################################################
# In this case, the class labels are stored in an array with 0's in the male
# samples and 1's in the positions with female ones.

print(y)

############################################################################
# We can split the dataset using the sklearn function
# :func:`~sklearn.model_selection.train_test_split`.
#
# The function will return two
# :class:`~skfda.representation.grid.FDataGrid`'s, ``X_train`` and ``X_test``
# with the corresponding partitions, and arrays with their class labels.

X_train, X_test, y_train, y_test = train_test_split(fd, y, test_size=0.25,
                                                    stratify=y, random_state=0)

############################################################################
# We will fit the classifier
# :class:`~skfda.ml.classification.KNeighborsClassifier`
# with the training partition. This classifier accepts as input a
# :class:`~skfda.representation.grid.FDataGrid`.

clf = MaximumDepthClassifier()
clf.fit(X_train, y_train)

############################################################################
# Once it is fitted, we can predict labels for the test samples.
#
# To predict the label of a test sample, the classifier will assign the
# sample to the class where it is deeper. See the documentation of the
# depths module for a list of available depths in
# :doc:`/modules/exploratory/depths`. By default modified band depth is used.

pred = clf.predict(X_test)

############################################################################
# The :func:`~skfda.ml.classification.MaximumDepthClassifier.score` method
# allows us to calculate the mean accuracy for the test data. In this case we
# obtained around 87.5% of accuracy.

score = clf.score(X_test, y_test)
print(score)
