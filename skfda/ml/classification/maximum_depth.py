"""Maximum depth for supervised classification."""
import numpy as np

from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted as sklearn_check_is_fitted

from skfda.exploratory.depth import *

class MaximumDepth(BaseEstimator, ClassifierMixin):
    """Maximum depth classifier for functional data.

    Test samples are classified to the class where they are deeper.

    Parameters
    ----------
    depth_method : callable, (default
        :class:`IntegratedDepth <skfda.depth.IntegratedDepth>`)
        The depth class to use when calculating the depth of a test 
        samples in a class. See the documentation of the depths module
        for a list of available depths. By default it is the one used
        by Fraiman and Muniz.
    Examples
    --------
    Firstly, we will import and split the Berkeley Growth Study dataset

    >>> from skfda.datasets import fetch_growth
    >>> from sklearn.model_selection import train_test_split
    >>> dataset = fetch_growth()
    >>> fd = dataset['data']
    >>> y = dataset['target']
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     fd, y, test_size=0.25, stratify=y, random_state=0)

    We will fit a Maximum depth classifier

    >>> from skfda.ml.classification import MaximumDepth
    >>> clf = MaximumDepth()
    >>> clf.fit(X_train, y_train)
    MaximumDepth()

    We can predict the class of new samples

    >>> clf.predict(X_test) # Predict labels for even samples
    array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0,
           1, 1])

    See also
    --------
    :class:`~skfda.ml.classification.DD-plot`

    """

    def __init__(self, depth_method=IntegratedDepth()):
        """Initialize the classifier."""
        self.depth_method = depth_method

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values.

        Args:
            X (:class:`FDataGrid`, array_matrix): Training data. FDataGrid
                with the training data or array matrix with shape
                [n_samples, n_samples] if metric='precomputed'.
            y (array-like or sparse matrix): Target values of
                shape = [n_samples] or [n_samples, n_outputs].

        """
        check_classification_targets(y)

        le = LabelEncoder()
        y_ind = le.fit_transform(y)
        self.classes_ = classes = le.classes_
        n_classes = classes.size
        if n_classes < 2:
            raise ValueError(f'The number of classes has to be greater than'
                             f' one; got {n_classes} class')

        self.distributions_ = [None]*n_classes
        for cur_class in range(0, n_classes):
            distribution = clone(self.depth_method).fit(X[y_ind == cur_class])
            self.distributions_[cur_class] = distribution

        return self
  
    def predict(self, X):
        """Predict the class labels for the provided data.

        Args:
            X (:class:`FDataGrid`): FDataGrid with the test samples.

        Returns:
            (np.array): y : array of shape [n_samples] or
            [n_samples, n_outputs] with class labels for each data sample.

        """
        sklearn_check_is_fitted(self)

        depths = [distribution.predict(X) for distribution in self.distributions_]
        return self.classes_[np.argmax(depths, axis=0)]
