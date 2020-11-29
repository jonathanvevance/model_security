import random
import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import inv
from numpy import identity as I

from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import PolynomialFeatures


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def plot_decision_boundary(X_train, y_train, clf):
    fig, ax = plt.subplots()
    X0, X1 = X_train[:, 0], X_train[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c = y_train, cmap = plt.cm.coolwarm, s = 20, edgecolors = 'k')
    plt.show()
    

def L2_dist(x1, x2):
    return np.sqrt(np.sum(np.square(x1 - x2)))


# Quadratic Programming Incremental SVM
class online_svm_qp():

    def __init__(self, true_clf, threshold = None, dist_DB = 1.0, degree = 1): 
        
        self.bias = None
        self.weights = None
        # self.X_retained = None
        self.X_retained = []  ## checking
        self.y_retained = None

        if hasattr(true_clf, "coef_"):
            self.true_weights = true_clf.coef_[0] 
            self.true_bias   = true_clf.intercept_[0]
        else:
            self.true_weights = None
            self.true_bias   = None

        self.dist_DB = dist_DB
        self.degree = degree # 
        self.support_vec_ids = None
        self.threshold = threshold
        self.clf = None # for plotting

        # buffer
        self.X_fit = None
        self.y_fit = np.array([])

        # new additions
        self.X_perm = []
        self.y_perm = []

    def fit(self, X, y):
        """
        inputs  : X_init (numpy array)
                  y_init (numpy vector)
        outputs : None 
        """

        if not isinstance(self.weights, np.ndarray): # hasnt been fit
            if not isinstance(self.X_fit, np.ndarray): # first call
                self.X_fit = X
            else:
                self.X_fit = np.vstack((self.X_fit, X))

            self.y_fit = np.append(self.y_fit, y)
            
            if (1 in self.y_fit) & (0 in self.y_fit):

                self.clf = SVC(kernel = 'linear')
                pf = PolynomialFeatures(self.degree, include_bias = False)
                self.X_fit = pf.fit_transform(self.X_fit)

                self.clf.fit(self.X_fit, self.y_fit)
                self.bias = self.clf.intercept_[0] 
                self.weights = self.clf.coef_[0]
                self.support_vec_ids = set(self.clf.support_)
                self.update_candidates(self.X_fit, self.y_fit)
                self.X_fit, self.y_fit = None, None

        else:            

            self.clf = SVC(kernel = 'linear')
            if self.X_perm != []:
                self.clf.fit(np.vstack((X, np.array(self.X_perm))), np.append(y, self.y_perm))
            else:
                self.clf.fit(X, y)

            self.bias = self.clf.intercept_[0] 
            self.weights = self.clf.coef_[0]
            self.support_vec_ids = set(self.clf.support_)
            self.update_candidates(X, y)


    def distance_DB(self, x, y):
        """
        inputs  : x (numpy vector)
                  y (int: 1 or -1)
        outputs : distance to hyperplane
        """
        assert (y == -1) or (y == 1) # should not pass 0
        fx = np.dot(self.weights, x) + self.bias
        return y * (fx) / np.linalg.norm(self.weights)

    
    def true_distance_DB(self, x):
        """
        inputs  : x (numpy vector)
        outputs : distance to true hyperplane
        """
        if isinstance(self.true_weights, np.ndarray):
            fx = abs(np.dot(self.true_weights, x) + self.true_bias)
            return fx / np.linalg.norm(self.true_weights)
        else:
            return np.inf

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

            if i < X.shape[0]:
                pos_dist = L2_dist(pos_centroid, X[i])
                neg_dist = L2_dist(neg_centroid, X[i])
            else:
                pos_dist = L2_dist(pos_centroid, self.X_perm[i - X.shape[0]])
                neg_dist = L2_dist(neg_centroid, self.X_perm[i - X.shape[0]])

            R_pos = min(pos_dist, R_pos)
            R_neg = min(neg_dist, R_neg)

        # centoids to hyperplane distances
        R0_pos = self.distance_DB(pos_centroid, 1)
        R0_neg = self.distance_DB(neg_centroid, -1)

        # selecting CSVs
        self.X_retained = []
        self.y_retained = []
        for i in range(X.shape[0]):

            if (self.true_distance_DB(X[i]) <= self.dist_DB):
                self.X_perm.append(X[i])
                self.y_perm.append(y[i])

            elif i in self.support_vec_ids:
                self.X_retained.append(X[i])
                self.y_retained.append(y[i])

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
        
        if not isinstance(self.weights, np.ndarray):
            self.fit(X, y)
            return

        if X.ndim == 1:
            X = np.expand_dims(X, axis = 0)
            y = np.expand_dims(y, axis = 0)

        pf = PolynomialFeatures(self.degree, include_bias = False)
        X = pf.fit_transform(X)

        fit_flag = False
        X_violations = []
        y_violations = []
        for i in range(X.shape[0]):

            if (self.true_distance_DB(X[i]) <= self.dist_DB):
                fit_flag = True
                self.X_perm.append(X[i])
                self.y_perm.append(y[i])

            elif y[i] == 1:
                if (self.distance_DB(X[i], 1) < 1 - tol):
                    fit_flag = True
                    X_violations.append(X[i])
                    y_violations.append(y[i])

            else:
                if (self.distance_DB(X[i], -1) < 1 - tol):
                    fit_flag = True
                    X_violations.append(X[i])
                    y_violations.append(y[i])

        if fit_flag == False:
            return 

        else:
            if X_violations != []:

                if len(self.X_retained) == 0:
                    self.X_retained = np.array(X_violations)
                    self.y_retained = np.array(y_violations)
                else:
                    self.X_retained = np.vstack((self.X_retained, X_violations))
                    self.y_retained = np.append(self.y_retained, y_violations)

                self.fit(self.X_retained, self.y_retained)
            
            elif self.X_retained.shape[0] > 0:
                self.fit(self.X_retained, self.y_retained)

            else:
                self.clf = SVC(kernel = 'linear')
                self.clf.fit(self.X_perm, self.y_perm)

                self.bias = self.clf.intercept_[0] 
                self.weights = self.clf.coef_[0]
                self.support_vec_ids = set(self.clf.support_)


    def reduce_retained(self, max_size = 100):

        if len(self.X_perm) >= max_size:
            
            distances = []
            for i in range(len(self.X_per)):
                distances.append(self.true_distance_DB(self.X_perm[i, :]))

            self.X_perm = self.X_perm[np.argsort(distances)[: max_size]]
            self.X_retained = []
            self.y_retained = []

        else:

            distances = []
            max_size = int(max_size - len(self.X_perm))

            for i in range(len(self.X_retained)):
                distances.append(self.true_distance_DB(self.X_retained[i, :]))

            selected_idx = np.argsort(distances)[: max_size]
            self.X_retained = self.X_retained[selected_idx]
            self.y_retained = self.y_retained[selected_idx]


    def predict(self, X):

        if isinstance(self.weights, np.ndarray):
            if X.ndim == 1:
                X = np.expand_dims(X, axis = 0)
            
            pf = PolynomialFeatures(self.degree, include_bias = False)
            X = pf.fit_transform(X)
            return self.clf.predict(X)

        else:
            return np.array([random.choice([0, 1]) for __ in range(len(X))])


# Least Squares Incremental SVM
class online_lssvm():

    def __init__(self, rho = 0.001, degree = 1):
        
        self.C = None
        self.rho = rho
        self.bias = None
        self.weights = None
        self.degree = degree

    def fit(self, X, y):

        if X.ndim == 1:
            X = np.expand_dims(X, axis = 0)
            y = np.expand_dims(y, axis = 0)
        
        y = y.copy()
        y[y == 0] = -1 # labels must be -1 or 1
        X = PolynomialFeatures(self.degree).fit_transform(X)

        self.C = X.T @ inv(X @ X.T + self.rho * I(X.shape[0])) @ X # p X p
        weight_vec = (inv(X.T @ X + self.rho * I(X.shape[1])) @ X.T) @ y 
        self.bias, self.weights = weight_vec[0], weight_vec[1:]

    def update(self, X, y):
        
        if not isinstance(self.C, np.ndarray):
            self.fit(X, y)
            return

        if X.ndim == 1:
            X = np.expand_dims(X, axis = 0)
            y = np.expand_dims(y, axis = 0)

        r = self.rho
        N = X.shape[0]
        p = self.C.shape[0]
        
        y = y.copy()
        y[y == 0] = -1 # labels must be -1 or 1
        X = PolynomialFeatures(self.degree).fit_transform(X)

        self.C += (self.C - I(p)) @ X.T @ inv(r * I(N) - X @ (self.C - I(p)) @ X.T) @ X @ (self.C - I(p))
        weight_vec = np.insert(self.weights, 0, self.bias)
        weight_vec += (self.C - I(p)) @ X.T @ inv(r * I(N) - X @ (self.C - I(p)) @ X.T) @ (X @ weight_vec - y)
        self.bias, self.weights = weight_vec[0], weight_vec[1:]

    def predict(self, X):
        
        X = PolynomialFeatures(self.degree).fit_transform(X)
        weight_vec = np.insert(self.weights, 0, self.bias)
        return np.where(X @ weight_vec >= 0, 1, 0)


# Stochastic Gradient Descent SVM
class online_svm_sgd(SGDClassifier):
    def __init__(self, degree = 1):
        self.bias = None
        self.weights = None
        self.degree = degree
        super().__init__(loss = 'hinge')

        # buffer
        self.X_fit = None
        self.y_fit = np.array([])

    def fit(self, X, y):

        if not isinstance(self.weights, np.ndarray):
            if not isinstance(self.X_fit, np.ndarray): # first call
                self.X_fit = X
            else:
                self.X_fit = np.vstack((self.X_fit, X))

            self.y_fit = np.append(self.y_fit, y)

            if (1 in self.y_fit) & (0 in self.y_fit):
                
                pf = PolynomialFeatures(self.degree, include_bias = False)
                self.X_fit = pf.fit_transform(self.X_fit) 

                super().fit(self.X_fit, self.y_fit)
                self.weights = self.coef_[0]
                self.bias = self.intercept_[0]

        else:
            
            pf = PolynomialFeatures(self.degree, include_bias = False)
            X = pf.fit_transform(X)

            super().fit(X, y)
            self.weights = self.coef_[0]
            self.bias = self.intercept_[0]

    def update(self, X, y):
        
        if not isinstance(self.weights, np.ndarray):
            self.fit(X, y)
            return

        if X.ndim == 1:
            X = np.expand_dims(X, axis = 0)
            y = np.expand_dims(y, axis = 0)

        pf = PolynomialFeatures(self.degree, include_bias = False)
        X = pf.fit_transform(X)

        super().partial_fit(X, y)
        self.weights = self.coef_[0]
        self.bias = self.intercept_[0]

    def predict(self, X):

        if not isinstance(self.weights, np.ndarray):
            return np.array([random.choice([0, 1]) for __ in range(len(X))])

        else:
            pf = PolynomialFeatures(self.degree, include_bias = False)
            X = pf.fit_transform(X)
            return super().predict(X)
