from sklearn.base import BaseEstimator, ClusterMixin, ClassifierMixin, clone
# from sklearn.utils.metaestimators import if_delegate_has_method
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
        unique_previous = np.unique(previous_labels)
        unique_new = np.unique(new_labels)
        
        cost_matrix = np.zeros((len(unique_previous), len(unique_new)))
        
        for i, prev_label in enumerate(unique_previous):
            for j, new_label in enumerate(unique_new):
                cost_matrix[i, j] = np.sum((previous_labels == prev_label) & (new_labels == new_label))
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
        
        mapping = {unique_new[col_idx]: unique_previous[row_idx] 
                   for col_idx, row_idx in zip(col_ind, row_ind)}
        
        aligned_labels = np.array([mapping.get(label, label) for label in new_labels])
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
