import random
import numpy as np
import copy
import logging

import numpy as np

from sklearn import svm
from sklearn.datasets import load_svmlight_file

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from algorithms.OnlineBase import OnlineBase

class informed_attacker:
    
    def __init__(self, X, target, clf):
        
        self.range = []
        self.target = target # true clf
        self.started = False
        self.clf = clf

        for i in range(X.shape[1]):
            self.range.append((X[:, i].min(), X[:, i].max()))

        # store queries
        self.X = None
        self.Y = []

    def query_fit(self):
        
        x = []
        for min_Xi, max_Xi in self.range:
            x.append(random.uniform(min_Xi, max_Xi))
        x = np.array(x)

        if self.Y == []:
            self.X = np.expand_dims(x, axis = 0)
        else:
            self.X = np.vstack((self.X, x))
        
        x = np.expand_dims(x, axis = 0)
        y = self.target.predict(x)
        self.Y.append(y)

        if not self.started:
            if (1 in self.Y) and (0 in self.Y):
                self.started = True
        
        if self.started:
            self.clf.fit(self.X, self.Y)

        return x, y

    def predict(self, X):

        if self.started:
            return self.clf.predict(X)
        else:
            return np.array([random.choice([0, 1]) for __ in range(len(X))])


class benign_user:
    
    def __init__(self, X, target, clf, width = 3.0):
        
        self.range = []
        self.target = target # true clf
        self.started = False
        self.clf = clf

        for i in range(X.shape[1]):
            mid = 0.5 * (X[:, i].max() + X[:, i].min())
            feature_width = (width/2) * (X[:, i].max() - X[:, i].min())
            self.range.append((mid - feature_width, mid + feature_width))

        # store queries
        self.X = None
        self.Y = []

    def query_fit(self):
        
        x = []
        for min_Xi, max_Xi in self.range:
            x.append(random.uniform(min_Xi, max_Xi))
        x = np.array(x)

        if self.Y == []:
            self.X = np.expand_dims(x, axis = 0)
        else:
            self.X = np.vstack((self.X, x))
        
        x = np.expand_dims(x, axis = 0)
        y = self.target.predict(x)
        self.Y.append(y)

        if not self.started:
            if (1 in self.Y) and (0 in self.Y):
                self.started = True
        
        if self.started:
            self.clf.fit(self.X, self.Y)

        return x, y

    def predict(self, X):

        if self.started:
            return self.clf.predict(X)
        else:
            return np.array([random.choice([0, 1]) for __ in range(len(X))])

