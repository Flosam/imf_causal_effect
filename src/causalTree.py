import numpy as np

class Node:
    def __init__(self, depth, tau=0.0, feature=None, threshold=None, left=None, right=None, n_samples=0):
        self.tau = tau
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.n_samples = n_samples
        self.depth = depth
        self.is_leaf = False
        

class CausalTree:
    def __init__(self, max_depth=3, min_sample_leaf=10, val_split=0.5):
        self.max_depth = max_depth
        self.min_sample_leaf = min_sample_leaf
        self.val_split = val_split
        self.root = None

    def fit(self, X, y, w):
        # data splitting between training and estimation samples
        N = len(y)
        idx = np.random.permutation(N)
        idx_tr, idx_est = idx[:N//2], idx[N//2:]

        x_tr, y_tr, w_tr = X[idx_tr], y[idx_tr], w[idx_tr]
        x_est, y_est, w_est = X[idx_est], y[idx_est], w[idx_est]

        # data splitting between training and validation samples
        N_tr = len(y_tr)
        idx_val = np.random.permutation(N_tr)
        idx_tr_tr, idx_tr_val = idx_val[:N_tr//2], idx_val[N_tr//2:]
        x_tr_tr, y_tr_tr, w_tr_tr = x_tr[idx_tr_tr], y_tr[idx_tr_tr], w_tr[idx_tr_tr]
        x_tr_val, y_tr_val, w_tr_val = x_tr[idx_tr_val], y_tr[idx_tr_val], w_tr[idx_tr_val]

        # compute global propensity for training (used in honest criterion)
        p_tr = np.mean(w_tr_tr)
        p_val = np.mean(w_tr_val)

        # Build tree using training data
        self.root = self.build_tree(x_tr_tr, y_tr_tr, w_tr_tr, len(y_tr_tr), p_tr, depth=0, max_depth=self.max_depth, min_leaf=self.min_sample_leaf)

        # Prune tree using validation data
        self.root = self.prune(self.root, x_tr_val, y_tr_val, w_tr_val, len(y_tr_val), p_val)

        # Honest estimation of treatment effects in leaves using estimation data
        self.estimate_honest_values(self.root, x_est, y_est, w_est)

    def predict(self, X):
        return np.array([self.traverse_tree(x, self.root) for x in X])
    
    def traverse_tree(self, x, node):
        if node.is_leaf:
            return node.tau_hat
        if x[node.feature] <= node.treshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)
    
    def prune(self, node, x_val, y_val, w_val, n_val, p):
        # recursive pruning that looks at leafs first and goes up the tree
        if node.is_leaf:
            return node
        
        # send to children
        idx_left = x_val[:, node.feature] <= node.threshold
        node.left = self.prune(node.left, x_val[idx_left], y_val[idx_left], w_val[idx_left], n_val, p)
        node.right = self.prune(node.right, x_val[~idx_left], y_val[~idx_left], w_val[~idx_left], n_val, p)

        # If both children are leaves, check if split helps on validation data
        if node.left.is_leaf and node.right.is_leaf:
            score_left = self.calculate_honest_criterion(y_val[idx_left], w_val[idx_left], n_val, p)
            score_right = self.calculate_honest_criterion(y_val[~idx_left], w_val[~idx_left], n_val, p)
            score_split = score_left + score_right
            score_nosplit = self.calculate_honest_criterion(y_val, w_val, n_val, p)

            if score_nosplit >= score_split:
                node.is_leaf = True
                node.left = node.right = None
                
        return node
        
        
    def calculate_honest_criterion(self, y, w, n_tr, p):
        # separate control and treated data
        y0 = y[w==0]
        y1 = y[w==1]
        n1 = len(y1)
        n0 = len(y0)

        if n1 < 2 or n0 < 2:
            return -np.inf
        
        # calculate treatment effect by difference of means
        tau_hat = np.mean(y1) - np.mean(y0)

        # calculate variance
        var1 = np.var(y1, ddof=1)
        var0 = np.var(y0, ddof=1)

        # term 1
        term1 = (1/n_tr) * (n1*tau_hat**2)

        # term 2
        term2 = (2/n_tr)*(var1/p + var0/(1-p))

        return term1 - term2
    
    def find_best_split(self, X, y, w, n_tr, p, min_leaf):
        best_score = -np.inf
        best_feature = None
        n_features = X.shape[1]

        for j in range(n_features):
            thresholds = np.unique(X[:, j])
            for threshold in thresholds:
                left_mask = X[:, j] <= threshold
                right_mask = X[:, j] > threshold

                if np.sum(left_mask) < min_leaf or np.sum(right_mask) < min_leaf:
                    continue

                score_left = self.calculate_honest_criterion(y[left_mask], w[left_mask], n_tr, p)
                score_right = self.calculate_honest_criterion(y[right_mask], w[right_mask], n_tr, p)
                
                if score_left == -np.inf or score_right == -np.inf:
                    continue
                
                score = score_left + score_right

                if score > best_score:
                    best_score = score
                    best_feature = (j, threshold)

        return best_feature
    
    def build_tree(self, x_tr, y_tr, w_tr, n_tr, p, depth, max_depth, min_leaf):
        node = Node(depth)

        if depth >= max_depth:
            node.is_leaf = True
            return node
        
        split = self.find_best_split(x_tr, y_tr, w_tr, n_tr, p, min_leaf)

        if split == None:
            node.is_leaf = True
            return node

        node.feature, node.threshold = split

        # get indices from threshold
        idx_left = x_tr[:, node.feature] <= node.threshold
        # create left and right children
        node.left = self.build_tree(x_tr[idx_left], y_tr[idx_left], w_tr[idx_left], n_tr, p, depth+1, max_depth, min_leaf)
        node.right = self.build_tree(x_tr[~idx_left], y_tr[~idx_left], w_tr[~idx_left], n_tr, p, depth+1, max_depth, min_leaf)

        return node
    
    def estimate_honest_values(self, node, x_est, y_est, w_est):
        if node.is_leaf:
            # calculate CATE using estiamtion sample
            y1 = x_est[w_est==1]
            y0 = x_est[w_est==0]   

            if len(y1) > 0 and len(y0) > 0:
                node.tau_hat = np.mean(y1) - np.mean(y0)
            else:
                node.tau_hat = 0.0
            return 
        
        # if not leaf, pass down to children
        idx_left = x_est[:, node.feature] <= node.threshold
        self.estimate_honest_values(node.left, x_est[idx_left], y_est[idx_left], w_est[idx_left])
        self.estimate_honest_values(node.right, x_est[~idx_left], y_est[~idx_left], w_est[~idx_left])

    
    
