import random
import numpy as np

class naive_attacker:
    
    def __init__(self, target, clf, x1_range = (0, 20), x2_range = (-5, 15)):
        
        self.x1_range, self.x2_range = x1_range, x2_range
        self.target = target # true clf
        self.started = False
        self.clf = clf

        # store queries
        self.X = None
        self.Y = []

    def query_fit(self):
        
        x1 = random.randint(self.x1_range[0], self.x1_range[1])
        x2 = random.randint(self.x2_range[0], self.x2_range[1])
        x = np.array([x1, x2])

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

