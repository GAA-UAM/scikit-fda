"""
K-nearest neighbors classification
==================================

Shows the usage of the k-nearest neighbors classifier.
"""

# Author: Pablo Marcos Manchón
# License: MIT

import skfda
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from skfda.ml.classification import KNeighborsClassifier


################################################################################
#
# Text
#
#

data = skfda.datasets.fetch_growth()
X = data['data']
y = data['target']

X[y==0].plot(color='C0')
X[y==1].plot(color='C1')

################################################################################
#
#
#
# Text

print(y)

################################################################################
#
#
#
# Text

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


################################################################################
#
#
#
# Text

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

################################################################################
#
#
#
# Text


pred = knn.predict(X_test)
print(pred)

################################################################################
#
#
#
# Text

score = knn.score(X_test, y_test)
print(score)

################################################################################
#
#
#
# Text

probs = knn.predict_proba(X_test[:5])
print(probs)


################################################################################
#
#
#
# Text

param_grid = {'n_neighbors': np.arange(1, 12, 2)}


knn = KNeighborsClassifier()
gscv = GridSearchCV(knn, param_grid, cv=KFold(shuffle=True, random_state=0))
gscv.fit(X, y)


print("Best params:", gscv.best_params_)
print("Best score:", gscv.best_score_)


################################################################################
#
#
#
# Text


plt.figure()
plt.bar(param_grid['n_neighbors'], gscv.cv_results_['mean_test_score'])

plt.xticks(param_grid['n_neighbors'])
plt.ylabel("Number of Neighbors")
plt.xlabel("Test score")
plt.ylim((0.9, 1))


################################################################################
#
#
#
# Text


knn = KNeighborsClassifier(metric='euclidean', sklearn_metric=True)
gscv2 = GridSearchCV(knn, param_grid, cv=KFold(shuffle=True, random_state=0))
gscv2.fit(X, y)

print("Best params:", gscv2.best_params_)

################################################################################
#
#
#
# Text

print("Mean score time (seconds)")
print("L2 distance:", np.mean(gscv.cv_results_['mean_score_time']), "(s)")
print("Sklearn distance:", np.mean(gscv2.cv_results_['mean_score_time']), "(s)")

################################################################################
#
#
#
# Text

plt.show()
