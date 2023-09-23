from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from collections import Counter
import numpy as np

class DecisionTreeClassifier(BaseEstimator, TransformerMixin, ClassifierMixin):
    class Node:
        '''
        feature = feature index which represents each column of numpy matrix
        threshold = value to be compared to new incoming data, this one could be a mean(numeric) or 1 (categoric)
        dtype_feature = data type of the feature
        partition_metric = gini or gain (entropy) which splitted the matrix
        n_samples = number of samples per partition in a node
        value = categoric list representation of value counts of classes, e.g [50, 60] for 0 and 1
        predicted_label = if the node is a leaf, this is the class it predicts
        left = left branch
        right = right branch
        depth = dpeth in which the node is located from root (depth = 0)
        '''
        def __init__(self, feature = None, threshold = None, dtype_feature = None ,
                     partition_metric = None, n_samples = None,
                    value = None, predicted_label = None, left = None, right = None, depth = None):
            self.feature = feature
            self.threshold = threshold
            self.dtype_feature = dtype_feature
            self.partition_metric = partition_metric
            self.n_samples = n_samples
            self.value = value
            self.predicted_label = predicted_label
            self.left = left
            self.right = right
            self.depth = depth

        def is_leaf(self):
            return self.left is None and self.right is None

        def is_internal(self):
            return not self.is_leaf()
        
    class Tree:
        def __init__(self, root_node):
            self.root = root_node

        def depth(self):
            return self._calculate_depth(self.root)

        def num_nodes(self):
            return self._count_nodes(self.root)

        def num_leaves(self):
            return self._count_leaves(self.root)

        def _calculate_depth(self, node):
            if node is None:
                return -1
            left_depth = self._calculate_depth(node.left)
            right_depth = self._calculate_depth(node.right)
            return max(left_depth, right_depth) + 1

        def _count_nodes(self, node):
            if node is None:
                return 0
            left_count = self._count_nodes(node.left)
            right_count = self._count_nodes(node.right)
            return left_count + right_count + 1

        def _count_leaves(self, node):
            if node is None:
                return 0
            if node.is_leaf():
                return 1
            left_leaves = self._count_leaves(node.left)
            right_leaves = self._count_leaves(node.right)
            return left_leaves + right_leaves
 
    def __init__(self, criterion = 'gini', max_depth = None, min_samples_split = 2,
                min_samples_leaf = 1, max_features = None, random_state = None, ccp_alpha = 0.0):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.ccp_alpha = ccp_alpha
    
    def _init_matrix(self, X, y):
        # converting into numpy arrays
        X, y = np.array(X), np.array(y)
        
        # concatenating for a while
        data = np.column_stack((X, y))

        # Finding unique indices based on X rows
        unique_indices = np.unique(data[:, :-1] , axis = 0, return_index = True)[1]

        # selecting X and y that are unique
        X = X[unique_indices]
        y = y[unique_indices]
        return X, y
    
    def _gini(self, y_target):
        target_distribution = np.unique(y_target, return_counts = True)[1] / len(y_target)
        return 1 - (target_distribution ** 2).sum()
    
    def _is_categoric(self, X_feature):
        return np.all((X_feature == 0) | (X_feature == 1))
    
    def _get_data_type(self, X):
        return {n_feature: 'cat' if self._is_categoric(X[:, n_feature]) else 'num'
               for n_feature in range(X.shape[1])}
    
    def _numeric_to_categoric(self, X_feature, mean_value):
        return (X_feature <= mean_value) * 1
    
    def _get_unique_values(self, y):
        return np.unique(y, return_counts = True)[1].tolist()
    
    def _generate_split_params(self, X_train, y_train):
        # According to max_features the model uses to split matrix
        num_features = int(X_train.shape[1])
        if self.max_features is None:
            max_features = num_features
        elif isinstance(self.max_features, int):
            max_features = min(self.max_features, num_features)
        elif isinstance(self.max_features, float):
            max_features = max(int(self.max_features * num_features), 1)
        else:
            raise ValueError("Invalid value for max_features")
        np.random.seed(self.random_state)
        selected_feature_indices = np.random.choice(num_features, max_features, replace = False)
        total_metrics = list()
        
        # Then we iterate across selected feature
        for n_feature in selected_feature_indices:
            X_feature = X_train[:, n_feature].copy()
            if self.data_types[n_feature] == 'cat':
                left_branch_y = y_train[X_feature == 1].copy()
                right_branch_y = y_train[X_feature == 0].copy()
                n_left = len(left_branch_y)
                n_right = len(right_branch_y)
                left_metric = self._gini(left_branch_y)
                right_metric = self._gini(right_branch_y)
                result_metric = (n_left / len(y_train)) * left_metric + (n_right / len(y_train)) * right_metric
                total_metrics.append((1, result_metric, n_feature, 'cat'))
            else:
                total_metric_num = list()
                sorted_indices = np.argsort(X_feature)
                X_feature_sorted = np.unique(X_feature[sorted_indices].copy())
                y_train_sorted = y_train[sorted_indices].copy()
                mean_values = np.unique((X_feature_sorted[:-1] + X_feature_sorted[1:]) / 2)
                # in case X_feature_sorted has one single value
                mean_values = mean_values if len(mean_values) != 0 else X_feature_sorted
                for mean_value in mean_values:
                    X_feature_cat = self._numeric_to_categoric(X_feature, mean_value)
                    left_branch_y = y_train[X_feature_cat == 1].copy()
                    right_branch_y = y_train[X_feature_cat == 0].copy()
                    n_left = len(left_branch_y)
                    n_right = len(right_branch_y)
                    left_metric = self._gini(left_branch_y)
                    right_metric = self._gini(right_branch_y)
                    result_metric = (n_left / len(y_train)) * left_metric + (n_right / len(y_train)) * right_metric
                    total_metric_num.append((mean_value, result_metric, n_feature, 'num'))
                total_metrics.append(min(total_metric_num, key = lambda x: x[1]))
        return min(total_metrics, key = lambda x: x[1])
    
    def _split_dataframe(self, X, y, split_threshold_value, n_feature, dtype_feature):
        X_splitted_left = X[X[:, n_feature] <= split_threshold_value].copy() if self.data_types[n_feature] == 'num' else X[X[:, n_feature] == 1].copy()
        y_splitted_left = y[X[:, n_feature] <= split_threshold_value].copy() if self.data_types[n_feature] == 'num' else y[X[:, n_feature] == 1].copy()

        X_splitted_right = X[X[:, n_feature] > split_threshold_value].copy() if self.data_types[n_feature] == 'num' else X[X[:, n_feature] == 0].copy()
        y_splitted_right = y[X[:, n_feature] > split_threshold_value].copy() if self.data_types[n_feature] == 'num' else y[X[:, n_feature] == 0].copy()

        return X_splitted_left, X_splitted_right, y_splitted_left, y_splitted_right
    
    def _build_tree(self, X, y, depth = 0, parent_node = None):
        
        # Establishing terminating terms
        #print('Vector y')
        #print(y)
        #print('shape')
        #print(y.shape)
        #print('Vector X')
        #print(X)
        #print('shape')
        #print(X.shape)
        #print('Unique values')
        #print(np.unique(y))
        if len(y) == 0:
            if parent_node is not None:
                return parent_node
            else:
                return self.Node(n_samples = len(y),
                                 partition_metric = 0,
                                 value = self._get_unique_values(y),
                                 predicted_label = Counter(y).most_common()[0][0])
        
        if self.max_depth is not None and depth >= self.max_depth:
            return self.Node(n_samples = len(y),
                             partition_metric = 0,
                             value = self._get_unique_values(y),
                             predicted_label = Counter(y).most_common()[0][0])
        
        if isinstance(self.min_samples_split, float):
            if self.min_samples_split <= 1.0:
                min_samples_split = max(int(np.ceil(self.min_samples_split * len(y))), 2)
            else:
                min_samples_split = max(int(self.min_samples_split), 2)
        else:
            min_samples_split = self.min_samples_split
        
        if isinstance(self.min_samples_leaf, float):
            min_samples_leaf = max(int(np.ceil(self.min_samples_leaf * len(y))), 1)
        else:
            min_samples_leaf = max(self.min_samples_leaf, 1)

        if len(y) < min_samples_split or len(y) < min_samples_leaf:
            return self.Node(n_samples = len(y),
                 partition_metric = 0,
                 value = self._get_unique_values(y),
                 predicted_label = Counter(y).most_common()[0][0])
    
        # Minimum terminating condition needed, among the above ones, otherwise the recursion'd blow up
        if len(np.unique(y)) == 1:
            return self.Node(n_samples = len(y),
                 partition_metric = 0,
                 value = self._get_unique_values(y),
                 predicted_label = Counter(y).most_common()[0][0])
        
        initial_impurity = self._gini(y)

        # Continue building the tree as before, but check ccp_alpha before splitting
        if self.ccp_alpha > 0.0 and initial_impurity  <= self.ccp_alpha:
            # If ccp_alpha is reached or exceeded, create a leaf node
            return self.Node(n_samples = len(y),
                             partition_metric = initial_impurity ,
                             value = self._get_unique_values(y),
                             predicted_label = Counter(y).most_common()[0][0])
        # Splitting Node into two
        split_threshold_value, metric_value, n_feature, dtype_feature = self._generate_split_params(X, y)
        X_splitted_left, X_splitted_right, y_splitted_left, y_splitted_right = self._split_dataframe(X, y, 
                                                                                          split_threshold_value,
                                                                                          n_feature,
                                                                                         dtype_feature)
        # Creating branches
        depth += 1
        left_subtree = self._build_tree(X_splitted_left, y_splitted_left, depth, parent_node = self.Node(feature = n_feature,
                                                                                                    threshold = split_threshold_value,
                                                                                                    dtype_feature = dtype_feature,
                                                                                                    partition_metric = metric_value,
                                                                                                    n_samples = len(y),
                                                                                                    value = self._get_unique_values(y),
                                                                                                    predicted_label = Counter(y).most_common()[0][0],
                                                                                                    left = None,
                                                                                                    right = None,
                                                                                                    depth = depth))
        
        right_subtree = self._build_tree(X_splitted_right, y_splitted_right, depth, parent_node = self.Node(feature = n_feature,
                                                                                                    threshold = split_threshold_value,
                                                                                                    dtype_feature = dtype_feature,
                                                                                                    partition_metric = metric_value,
                                                                                                    n_samples = len(y),
                                                                                                    value = self._get_unique_values(y),
                                                                                                    predicted_label = Counter(y).most_common()[0][0],
                                                                                                    left = None,
                                                                                                    right = None,
                                                                                                    depth = depth))
        

        # Creating new nodes
        return self.Node(feature = n_feature,
                        threshold = split_threshold_value,
                        dtype_feature = dtype_feature,
                        partition_metric = metric_value,
                        n_samples = len(y),
                        value = self._get_unique_values(y),
                        predicted_label = Counter(y).most_common()[0][0],
                        left = left_subtree,
                        right = right_subtree,
                        depth = depth)
    
    def fit(self, X, y):
        self.X , self.y = self._init_matrix(X, y)
        self.data_types = self._get_data_type(self.X)
        self.root_node = self._build_tree(self.X, self.y)
        return self
    
    def predict(self, X):
        # initilize an empty list that stores all predictions of X
        y_pred = []
        
        # iterating on each row of the X matrix
        for row in X:
            current_node = self.root_node
            while not (current_node.is_leaf() or current_node is None):
                feature_index = current_node.feature
                threshold = current_node.threshold
                dtype_feature = current_node.dtype_feature
                
                if dtype_feature == 'cat':
                    if row[feature_index] == threshold:
                        current_node = current_node.left
                    else:
                        current_node = current_node.right
                else:
                    if row[feature_index] <= threshold:
                        current_node = current_node.left
                    else:
                        current_node = current_node.right
            y_pred.append(current_node.predicted_label)
            
        return np.array(y_pred)
    
    @property
    def tree_(self):
        return self.Tree(self.root_node)

    @property
    def feature_importances_(self):
        if not hasattr(self, 'root_node'):
            raise ValueError("This DecisionTreeClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")

        importances = np.zeros(self.X.shape[1])
        self._compute_feature_importances(self.root_node, importances)

        # Normalize the importances to sum up to 1
        importances /= importances.sum()

        return importances

    def _compute_feature_importances(self, node, importances):
        if node.is_internal():
            left_weight = node.left.n_samples / node.n_samples
            right_weight = node.right.n_samples / node.n_samples
            reduction = node.n_samples * node.partition_metric - (left_weight * node.left.n_samples * node.left.partition_metric + right_weight * node.right.n_samples * node.right.partition_metric)

            importances[node.feature] += reduction

        if node.left:
            self._compute_feature_importances(node.left, importances)
        if node.right:
            self._compute_feature_importances(node.right, importances)
            
    def cost_complexity_pruning_path(self, X, y, sample_weight=None):
        # Convert input to numpy arrays
        X , y = self._init_matrix(X, y)
        self.data_types = self._get_data_type(X)
        # Creating root node
        root_node = self._build_tree(X, y)
        
        # Initialize variables to store the pruning path
        ccp_alphas = []
        impurities = []
        
        # Calculate the initial impurity (before pruning)
        initial_impurity = self._gini(y)
        
        # Traverse the tree and compute pruning path
        self._compute_pruning_path(root_node, initial_impurity, ccp_alphas, impurities, X, y, sample_weight)
        
        return np.array(ccp_alphas), np.array(impurities)
    
    def _compute_pruning_path(self, node, impurity, ccp_alphas, impurities, X, y, sample_weight=None):
        if node.is_internal():
            left_weight = node.left.n_samples / node.n_samples
            right_weight = node.right.n_samples / node.n_samples
            reduction = node.n_samples * impurity - (left_weight * node.left.n_samples * node.left.partition_metric + right_weight * node.right.n_samples * node.right.partition_metric)

            if sample_weight is not None:
                reduction /= sum(sample_weight)

            alpha = reduction / (node.n_samples - 1)  # alpha is the cost-complexity factor

            # Check if alpha is greater than 0 (i.e., there's potential for pruning)
            if alpha > 0:
                ccp_alphas.append(alpha)
                impurities.append(impurity)

            # Continue with left and right subtrees
            self._compute_pruning_path(node.left, impurity, ccp_alphas, impurities, X, y, sample_weight)
            self._compute_pruning_path(node.right, impurity, ccp_alphas, impurities, X, y, sample_weight)