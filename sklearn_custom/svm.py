from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, sigmoid_kernel, polynomial_kernel, laplacian_kernel
from cvxopt import matrix, solvers
import numpy as np

solvers.options['show_progress'] = False

class SVC(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, kernel = 'rbf', C = 1.5, degree = 3, gamma = None, coef0 = 0.0, 
                 probability = False, random_state = None):
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.probability = probability
        self.random_state = random_state
        
    def _init_X(self, X):
        return np.array(X, dtype = 'float')
    
    def _init_y(self, y):
        return np.where(y == 0, -1, y)
    
    def _initialize_weights(self, input_size):
        np.random.seed(self.random_state)
        limit = np.sqrt(6 / (input_size + 1))
        return np.random.uniform(-limit, limit, size = (input_size,))
    
    # Kernels to transform X matrix
    def _rbf_kernel(self, X, Y):
        return rbf_kernel(X = X, Y = Y, gamma = self.gamma)
    
    def _linear_kernel(self, X, Y):
        return linear_kernel(X = X, Y = Y)
    
    def _poly_kernel(self, X, Y):
        return polynomial_kernel(X = X, Y = Y, degree = self.degree, gamma = self.gamma, coef0 = self.coef0)
    
    def _sigmoid_kernel(self, X, Y):
        return sigmoid_kernel(X = X, Y = Y, gamma = self.gamma, coef0 = self.coef0)
    
    def _laplacian_kernel(self, X, Y):
        return laplacian_kernel(X = X, Y = Y, gamma = self.gamma)
        
    def _get_alphas(self, X, y, C):
        n, m = X.shape
        y = y.astype(float)
        K = getattr(self, f"_{self.kernel}_kernel")(X, X)
        P = matrix(C * np.outer(y,y) * K)
        if np.iscomplexobj(P):
            raise ValueError("Complex data not supported")
        q = matrix(-np.ones(n))
        if self.C is None:
            G = matrix(-np.eye(n))
            h = matrix(np.zeros(n))
        else:
            tmp1 = -np.eye(n)
            tmp2 = np.identity(n)
            G = matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n)
            tmp2 = np.ones(n) * C
            h = matrix(np.hstack((tmp1, tmp2)))
        A = matrix(y.reshape(1, -1))
        b = matrix(np.zeros(1))
        try:
            sol = solvers.qp(P, q, G, h, A, b)
        except Exception as e:
            print(e)
        alpha = np.array(sol['x'])
        return alpha.flatten(), K, sol
    
    def _get_bias(self, y, sv_alphas, sv_y, sv_K):
        return (y - np.matmul(sv_alphas * sv_y, sv_K)).mean()
    
    def fit(self, X, y):
        if len(X) == 0:
            raise ValueError('Array is empty')
        if np.iscomplexobj(X) or np.iscomplexobj(y):
            raise ValueError("Complex data not supported.")
            
        # init arrays
        self.X = self._init_X(X)
        self.y = self._init_y(y)
        try:
            # converge alphas and get K
            alphas, self.K, self.sol = self._get_alphas(self.X, self.y, self.C)
        except Exception as e:
            # if there's an error the kernel is changed to rbf
            print(f"Error kernel {self.kernel}", e)
            print("Changing kernel to rbf")
            self.kernel = 'rbf' 
            alphas, self.K, self.sol = self._get_alphas(self.X, self.y, self.C)  
        
        # get support vectors
        sv_index = np.where(alphas > 1e-5)
        self.sv_alphas = alphas[sv_index]
        self.sv_x = self.X[sv_index]
        self.sv_y = self.y[sv_index]
        self.sv_K = self.K[sv_index]
        
        # get bias
        self.b = self._get_bias(y, self.sv_alphas, self.sv_y, self.sv_K)
        
        return self
    
    def transform(self, X):
        pass
    
    def get_bias(self):
        return self.b
    
    def get_support_vectors(self):
        return self.sv_alphas, self.sv_x, self.sv_y, self.sv_K
    
    def get_opt_solution(self):
        return self.sol
    
    def predict(self, X):
        # obtaining K values for new inputs
        X_k = getattr(self, f"_{self.kernel}_kernel")(self.sv_x, X)
        
        # getting predictions with bias
        y_pred = self.b + np.matmul(self.sv_alphas * self.sv_y, X_k)
        
        # returning in terms of 1 and 0
        return np.select([y_pred >= 0, y_pred < 0], [1, 0])
    
    def predict_proba(self, X):
        if self.probability:
            # obtaining K values for new inputs
            X_k = getattr(self, f"_{self.kernel}_kernel")(self.sv_x, X)

            # Obtaining probabilities
            raw_scores = self.b + np.matmul(self.sv_alphas * self.sv_y, X_k)
            probabilities = 1 / (1 + np.exp(-raw_scores))

            # Returning negative and  positives proabilities
            return np.column_stack((1 - probabilities, probabilities))
        else:
            raise ValueError('You have to set Probability to True')
            
    def predict_log_proba(self, X):
        if self.probability:
            # obtaining K values for new inputs
            X_k = getattr(self, f"_{self.kernel}_kernel")(self.sv_x, X)

            # Obtaining probabilities
            raw_scores = self.b + np.matmul(self.sv_alphas * self.sv_y, X_k)
            probabilities = 1 / (1 + np.exp(-raw_scores))

            # Returning negative and  positives log_probabilities
            return np.column_stack((np.log(1 - probabilities), np.log(probabilities)))
        else:
            raise ValueError('You have to set Probability to True')