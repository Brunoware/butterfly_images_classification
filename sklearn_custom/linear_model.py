from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
import numpy as np

class LogisticRegression(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, learning_rate = 0.001, max_iter = 10000, threshold = 0.5, tol = 1e-15, 
                 penalty = None, lam = None, gamma = None, random_state = None):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.threshold = threshold
        self.tol = tol
        self.penalty = penalty
        self.lam = lam
        self.gamma = gamma
        self.random_state = random_state

        
    def _add_bias(self, X):
        return np.hstack((np.ones((X.shape[0], 1)), X))
    
    def _adapt_x_matrix(self, X):
        X = np.array(X)
        X = self._add_bias(X)
        return X
    
    def _adapt_y_matrix(self, y):
        return np.array(y)
    
    def _adapt_df_matrix(self, X, y):
        X = self._adapt_x_matrix(X)
        y = self._adapt_y_matrix(y)
        return X, y
    
    def _initialize_weights(self, input_size):
        np.random.seed(self.random_state)
        limit = np.sqrt(6 / (input_size + 1))
        return np.random.uniform(-limit, limit, size = (input_size,))
    
    def _hyperplane(self, X, w):
        return np.dot(X, w)
    
    def _sigmoid(self, X, w):
        return 1 / (1 + np.exp(-self._hyperplane(X, w)))
    
    def _get_penalties(self, penalty):
        penalties = {'l1': lambda n, w, lam, gamma: (lam/ n) * np.sum(np.abs(w)),
                    'l2': lambda n, w, lam, gamma: (lam / n) * np.sum(w ** 2),
                    'elasticnet': lambda n, w, lam, gamma: gamma * (lam / n) * np.sum(np.abs(w)) +
                                                        (1 - gamma) * (lam / n) * np.sum(w ** 2),
                    None : lambda n, w, lam, gamma: 0}
        return penalties[penalty]
    
    def _get_penalties_derivatives(self, penalty):
        penalties = {'l1': lambda n, w, lam, gamma,: lam / n,
                    'l2': lambda n, w, lam, gamma: (2 * lam / n) * np.sum(w),
                    'elasticnet': lambda n, w, lam, gamma: gamma * (lam / n)  +
                                                        (1 - gamma) * (2 * lam / n) * np.sum(w),
                    None : lambda n, w, lam, gamma: 0}
        return penalties[penalty]
    
    def _loss_function(self, X, y, w, epsilon):
        sigmoid = np.clip(a = self._sigmoid(X, w), a_min = epsilon, a_max = 1 - epsilon)
        penalty_mode = self._get_penalties(self.penalty)(X.shape[0], w, self.lam, self.gamma)
        return (- 1 / X.shape[0]) * np.sum(y * np.log(sigmoid) + (1 - y) * np.log(1 - sigmoid)) + penalty_mode
    
    def _derivatives(self, X, y, w):
        y_pred = self._sigmoid(X, w)
        error = y - y_pred
        penalties = self._get_penalties_derivatives(self.penalty)(X.shape[0], w, self.lam, self.gamma)
        return (1 / X.shape[0]) * np.matmul(error, -X) + penalties
    
    def _change_parameters(self, w, derivatives, alpha):
        return w - alpha * derivatives
    
    def fit(self, X, y):
        # getting numpy arrays
        X, y = self._adapt_df_matrix(X, y)
        
        # initializing random weights
        self.weights = self._initialize_weights(X.shape[1])
        
        # iterations till converging
        for _ in range(self.max_iter):
            self.loss = self._loss_function(X, y, self.weights, self.tol)
            dw = self._derivatives(X, y, self.weights)
            self.weights = self._change_parameters(self.weights, dw, self.learning_rate)
            
        return self
    
    def transform(self , X, y = None):
        pass
    
    def predict(self, X, y = None):
        X = self._adapt_x_matrix(X)
        return (self._sigmoid(X, self.weights) > self.threshold) * 1
    
    def predict_proba(self, X, y = None):
        X = self._adapt_x_matrix(X)
        probabilities = self._sigmoid(X, self.weights)
        return np.column_stack([1 - probabilities, probabilities])
    
    def predict_log_proba(self, X, y = None):
        X = self._adapt_x_matrix(X)
        probabilities = self._sigmoid(X, self.weights)
        return np.column_stack([np.log(1 - probabilities), np.log(probabilities)])
    
    @property
    def get_loss_across_epochs(self):
        return self.loss