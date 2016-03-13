import os, time, sys, pdb
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle

import collections

class cwdmModel(object):
    def __init__(self, train_samp, test_samp, model_name=None, model=None,
                color_cut=None,color_label=None):

        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert hasattr(model, "score")
        if hasattr(model, "decision_function"):
            assert hasattr(model, "decision_function")
        else:
            assert hasattr(model, "predict_proba")

        # Pull out the colors from the training
        train_x = np.array(train_samp[color_cut[1][0]]-train_samp[color_cut[1][1]])
        train_y = np.array(train_samp[color_cut[0][0]]-train_samp[color_cut[0][1]])

        # Pull out the colors from the training
        test_x = np.array(test_samp[color_cut[1][0]]-test_samp[color_cut[1][1]])
        test_y = np.array(test_samp[color_cut[0][0]]-test_samp[color_cut[0][1]])

        # Set Attributes
        self.train_samp = train_samp
        self.test_samp = test_samp
        self.color_cut = color_cut
        self.color_label = color_label
        self.X_train = np.vstack([train_x,train_y]).T
        self.y_train = train_samp['iscwdm']
        self.X_test = np.vstack([test_x,test_y]).T
        self.y_test = test_samp['iscwdm']
        self.clf_name = model_name
        self.clf = model
        self.trained = None

    def train(self, X, y):
        # String serializion of the trained fit.
        # -- This is the only way I could get the fit to be preserved
        #    after pickle dumping/loading
        self.trained = pickle.dumps(self.clf.fit(X, y))

    def score(self, X, y):
        classifier = self.clf

        # Get the score
        score = classifier.score(X, y)

        return score
