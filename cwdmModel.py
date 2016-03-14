import os, time, sys, pdb
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle

import sklearn.base


class cwdmModel(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def __init__(self, X_train, y_train, X_test, y_test, model=None):

        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert hasattr(model, "score")
        if hasattr(model, "decision_function"):
            assert hasattr(model, "decision_function")
        else:
            assert hasattr(model, "predict_proba")

        # Store training data as attributes. Not the most efficient way...
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # Set Attributes
        self.clf = sklearn.base.clone(model)
        self.trained = None

    def fit(self, X, y):
        self.clf.fit(X,y)
        self.trained = True

    def predict(self, X, y=None):
        return self.clf.predict(X)
