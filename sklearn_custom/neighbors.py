from sklearn.neighbors import KDTree, BallTree
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
import numpy as np

class KNeighborsClassifier(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, n_neighbors = 5, weights = 'uniform', algorithm = 'auto', leaf_size = 30, p = 2,
                metric = 'minkwoski'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
    
    def _init_matrix(self, X, y):
        return np.array(X), np.array(y)
    
    def _calculate_weights(self, distances):
        if self.weights == 'uniform':
            return np.ones(len(distances))
        elif self.weights == 'distance':
            return 1 / (distances + 1e-6)
        elif callable(self.weights):
            return self.weights(distances)
        
    def _get_distance(self, x1, x2, metric):
        if metric == 'minkwoski':
            return np.sum(np.abs(x1 - x2) ** self.p) ** (1 / self.p)
        elif metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif metric == 'euclidean':
            return np.sum(np.abs(x1 - x2) ** 2) ** 1 / 2
        elif callable(metric):
            return metric(new_vector,current_vector)
    
    def _get_data_structure(self, X):
        if self.algorithm == 'auto':
            if len(X) > 1000:
                tree = BallTree(X, leaf_size = self.leaf_size)
            else:
                tree = KDTree(X, leaf_size = self.leaf_size)
        elif self.algorithm == 'ball_tree':
            tree = BallTree(X, leaf_size = self.leaf_size)
        elif self.algorithm == 'kd_tree':
            tree = KDTree(X, leaf_size = self.leaf_size)
        elif self.algorithm == 'brute_force':
            tree = BallTree(X, leaf_size = 1)
        return tree
        
    def fit(self, X, y):
        # only storing de X and y values that are going to be compared
        # to new incoming data points
        self.X, self.y = self._init_matrix(X, y)
        self.tree = self._get_data_structure(self.X)
        return self
    
    def predict(self, X):
        # it does it for each testing vector de matrix has
        return np.array([self._predict_vector(X_vector) for X_vector in X])
    
    def _predict_vector(self, x):
        # Calculating distances for a vector to each point of the training set
        distances, k_indices = self.tree.query([x], self.n_neighbors)

        # Getting the closest k neighbors
        k_nearest_labels = self.y[k_indices[0]]

        # Calculating weights
        weights = self._calculate_weights(distances[0])

        # Getting the most common classes
        most_common_votes = Counter(k_nearest_labels).most_common()

        # Return the most common class for the vector based on weights
        return max(most_common_votes, key = lambda item: np.sum(weights[k_nearest_labels == item[0]]))[0]