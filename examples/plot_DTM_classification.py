"""Distance to trimmed means (DTM) classification.

Shows the usage of DTM classifier.
"""

# Author: Pedro Martín Rodríguez-Ponga Eyriès
# License: MIT

import skfda
from skfda.ml.classification import DTMClassifier

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

############################################################################
# In this example we are going to show the usage of the distance to trimmed
# means classifier.
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
# :class:`~skfda.ml.classification.DTMClassifier`
# with the training partition. This classifier accepts as input a
# :class:`~skfda.representation.grid.FDataGrid`.

clf = DTMClassifier(proportiontocut=0.25)
clf.fit(X_train, y_train)

############################################################################
# Once it is fitted, we can predict labels for the test samples.
#
# To predict the label of a test sample, the classifier will assign the
# sample to the class to the class that minimizes the distance of
# the observation to the trimmed mean of the group. See the documentation of
# the depths module for a list of available depths in
# :doc:`/modules/exploratory/depths`. By default modified band depth and
# lp_distance are used.

pred = clf.predict(X_test)

############################################################################
# The :func:`~skfda.ml.classification.DTMClassifier.score` method
# allows us to calculate the mean accuracy for the test data. In this case we
# obtained around 87.5% of accuracy.

score = clf.score(X_test, y_test)
print(score)
