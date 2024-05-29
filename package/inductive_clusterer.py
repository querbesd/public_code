from sklearn.base import BaseEstimator, ClusterMixin, ClassifierMixin, clone
from sklearn.utils.metaestimators import if_delegate_has_method
from scipy.optimize import linear_sum_assignment
import numpy as np

class InductiveClusterer(BaseEstimator, ClusterMixin, ClassifierMixin):
    def __init__(self, clusterer, classifier, previous_labels=None):
        self.clusterer = clusterer
        self.classifier = classifier
        self.previous_labels = previous_labels

    def fit(self, X, y=None):
        self.clusterer_ = clone(self.clusterer)
        self.classifier_ = clone(self.classifier)
        y_pred = self.clusterer_.fit_predict(X)

        if self.previous_labels is not None:
            y_pred = self._align_labels(y_pred, self.previous_labels)

        self.classifier_.fit(X, y_pred)
        return self

    def predict(self, X):
        return self.classifier_.predict(X)

    def decision_function(self, X):
        return self.classifier_.decision_function(X)
    
    def _align_labels(self, new_labels, previous_labels):
        cost_matrix = np.zeros((len(np.unique(previous_labels)), len(np.unique(new_labels))))

        for i in range(len(np.unique(previous_labels))):
            for j in range(len(np.unique(new_labels))):
                cost_matrix[i, j] = np.sum((previous_labels == i) & (new_labels == j))

        row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
        mapping = {new_label: old_label for new_label, old_label in zip(col_ind, row_ind)}
        aligned_labels = np.vectorize(mapping.get)(new_labels)
        return aligned_labels

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.predict(X)

    def save_model(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
