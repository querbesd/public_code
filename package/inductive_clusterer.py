from sklearn.base import BaseEstimator, ClusterMixin, ClassifierMixin, clone
from sklearn.utils.metaestimators import if_delegate_has_method
from scipy.optimize import linear_sum_assignment
import numpy as np

import logging

logging.basicConfig(level=logging.INFO)

class InductiveClusterer(BaseEstimator, ClusterMixin, ClassifierMixin):
    def __init__(self, clusterer, classifier, previous_labels=None):
        self.clusterer = clusterer
        self.classifier = classifier
        self.previous_labels = previous_labels

    def fit(self, X, y=None):
        logging.info(f"Starting fit method. Data shape: {X.shape}")
        self.clusterer_ = clone(self.clusterer)
        self.classifier_ = clone(self.classifier)
        y_pred = self.clusterer_.fit_predict(X)
        logging.info(f"Cluster labels predicted: {y_pred}")

        if self.previous_labels is not None:
            y_pred = self._align_labels(y_pred, self.previous_labels)

        self.classifier_.fit(X, y_pred)
        return self

    def predict(self, X):
        logging.info(f"Starting predict method. Data shape: {X.shape}")
        return self.classifier_.predict(X)

    def decision_function(self, X):
        logging.info(f"Starting decision_function method. Data shape: {X.shape}")
        return self.classifier_.decision_function(X)
    
    def _align_labels(self, new_labels, previous_labels):
        logging.info("Aligning labels.")
        cost_matrix = np.zeros((len(np.unique(previous_labels)), len(np.unique(new_labels))))

        for i in range(len(np.unique(previous_labels))):
            for j in range(len(np.unique(new_labels))):
                cost_matrix[i, j] = np.sum((previous_labels == i) & (new_labels == j))

        row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
        mapping = {new_label: old_label for new_label, old_label in zip(col_ind, row_ind)}
        aligned_labels = np.vectorize(mapping.get)(new_labels)
        logging.info(f"Aligned labels: {aligned_labels}")
        return aligned_labels

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.predict(X)

    def save_model(self, path):
        logging.info(f"Saving model to {path}")
        with open(path, "wb") as f:
            joblib.dump(self, f)

    @classmethod
    def load_model(cls, path):
        logging.info(f"Loading model from {path}")
        with open(path, "rb") as f:
            return joblib.load(f)
