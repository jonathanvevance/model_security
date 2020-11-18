import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import inv
from numpy import identity as I
from sklearn import datasets
from sklearn.svm import SVC, LinearSVC              # LinearSVC is faster with large datasets
from sklearn.linear_model import SGDClassifier


def L2_dist(x1, x2):
    return np.sqrt(np.sum(np.square(x1 - x2)))


class online_svm_qp():

    def __init__(self, threshold = None):
        
        self.bias = None
        self.weights = None
        self.X_retained = None
        self.y_retained = None

        self.support_vec_ids = None
        self.threshold = threshold
        
        self.clf = None # for plotting

        # metrics to log 
        self.min_dist_DB = 1e10
        self.min_pairwise_dist = 1e10


    def fit(self, X, y):
        """
        inputs  : X_init (numpy array)
                  y_init (numpy vector)
        outputs : None 
        """
        self.clf = SVC(kernel = 'linear')
        self.clf.fit(X, y)
        self.bias = self.clf.intercept_ 
        self.weights = self.clf.coef_
        self.support_vec_ids = set(self.clf.support_)
        self.update_candidates(X, y)
        

    def distance_DB(self, x, y):
        """
        inputs  : x (numpy vector)
                  y (int: 1 or -1)
        outputs : distance to hyperplane
        """
        assert (y == -1) or (y == 1) # should not pass 0
        fx = np.dot(self.weights, x)[0] + self.bias[0]
        return y * (fx) / np.linalg.norm(self.weights)


    def update_candidates(self, X, y):
        """
        inputs  : X (numpy array)
                  y (numpy vector)
        outputs : None
        """
        # centroids
        pos_centroid = np.mean(X[np.where(y == 1)], axis = 0)
        neg_centroid = np.mean(X[np.where(y == 0)], axis = 0)

        # distance to closest svec
        R_pos = R_neg = 1e10
        for i in self.support_vec_ids: 
            
            pos_dist = L2_dist(pos_centroid, X[i])
            neg_dist = L2_dist(neg_centroid, X[i])

            R_pos = min(pos_dist, R_pos)
            R_neg = min(neg_dist, R_neg)

        # centoids to hyperplane distances
        R0_pos = self.distance_DB(pos_centroid, 1)
        R0_neg = self.distance_DB(neg_centroid, -1)

        # logging min_pairwise_dist (only searching among svecs)
        svec_ids_array = np.array(list(self.support_vec_ids))
        X_svecs_pos = X[svec_ids_array, :][y[svec_ids_array] == 1]
        X_svecs_neg = X[svec_ids_array, :][y[svec_ids_array] == 0]

        self.min_pairwise_dist = 1e5
        for x_pos in X_svecs_pos:
            for x_neg in X_svecs_neg:
                self.min_pairwise_dist = min(self.min_pairwise_dist, L2_dist(x_pos, x_neg))

        # selecting CSVs
        self.X_retained = []
        self.y_retained = []
        for i in range(X.shape[0]):

            if i in self.support_vec_ids: 
                self.X_retained.append(X[i])
                self.y_retained.append(y[i])

                # logging min_dist TO TRUTH (only need searching among support vecs)
                # self.min_dist_DB = min(self.min_dist_DB, CALL TRUE DB DISTANCE FUNCTION)

            else:
                if y[i] == 1:
                    d = L2_dist(X[i], pos_centroid)
                else:
                    d = L2_dist(X[i], neg_centroid)                

                if not isinstance(self.threshold, float):
                    if y[i] == 1:
                        if (d >= R_pos) and (self.distance_DB(X[i], 1) <= R0_pos):
                            self.X_retained.append(X[i])
                            self.y_retained.append(y[i])
                    else:
                        if (d >= R_neg) and (self.distance_DB(X[i], -1) <= R0_neg):
                            self.X_retained.append(X[i])
                            self.y_retained.append(y[i])

                else:                     
                    
                    L = 1 / np.linalg.norm(self.weights) # 1/2 margin distance
                    
                    if y[i] == 1:
                        t = d / (d + R_pos) + L / (L + self.distance_DB(X[i], 1)) - 1
                        if -self.threshold < t < self.threshold:
                            self.X_retained.append(X[i])
                            self.y_retained.append(y[i])
                    else:
                        t = d / (d + R_neg) + L / (L + self.distance_DB(X[i], -1)) - 1
                        if -self.threshold < t < self.threshold:
                            self.X_retained.append(X[i])
                            self.y_retained.append(y[i])
      
        self.X_retained = np.array(self.X_retained)
        self.y_retained = np.array(self.y_retained)

    def update(self, X, y, tol = 1e-3):

        if X.ndim == 1:
            X = np.expand_dims(X, axis = 0)
            y = np.expand_dims(y, axis = 0)

        X_violations = []
        y_violations = []
        for i in range(X.shape[0]):
            if y[i] == 1:
                if (self.distance_DB(X[i], 1) < 1 - tol):
                    X_violations.append(X[i])
                    y_violations.append(y[i])

            else:
                if (self.distance_DB(X[i], -1) < 1 - tol):
                    X_violations.append(X[i])
                    y_violations.append(y[i])

        if X_violations == []:
            return 

        else:

            self.X_retained = np.vstack((self.X_retained, X_violations))
            self.y_retained = np.append(self.y_retained, y_violations)

            self.fit(self.X_retained, self.y_retained)


class online_lssvm():

    def __init__(self, rho = 0.001):
        
        self.C = None
        self.rho = rho
        self.weights = None

    def fit(self, X, y):

        y[y == 0] = -1 # labels must be -1 or 1
        X = np.c_[np.ones((X.shape[0], 1)), X] # appending 1s

        self.C = X.T @ inv(X @ X.T + self.rho * I(X.shape[0])) @ X # p X p
        self.weights = (inv(X.T @ X + self.rho * I(X.shape[1])) @ X.T) @ y 
    
    def update(self, X, y):

        if X.ndim == 1:
            X = np.expand_dims(X, axis = 0)
            y = np.expand_dims(y, axis = 0)

        r = self.rho
        N = X.shape[0]
        p = self.C.shape[0]
        y[y == 0] = -1 # labels must be -1 or 1

        X = np.c_[np.ones((X.shape[0], 1)), X]
        self.C += (self.C - I(p)) @ X.T @ inv(r * I(N) - X @ (self.C - I(p)) @ X.T) @ X @ (self.C - I(p))
        self.weights += (self.C - I(p)) @ X.T @ inv(r * I(N) - X @ (self.C - I(p)) @ X.T) @ (X @ self.weights - y)

    def predict(self, X):
        
        X = np.c_[np.ones((X.shape[0], 1)), X]
        return np.where(X @ self.weights >= 0, 1, 0)


