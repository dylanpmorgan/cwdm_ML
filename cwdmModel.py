import os, time, sys, pdb
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle

import itertools
import sklearn.base

class cwdmModel(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def __init__(self,model=None, bands=None, model_params = {}):

        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert hasattr(model, "score")
        if hasattr(model, "decision_function"):
            assert hasattr(model, "decision_function")
        else:
            assert hasattr(model, "predict_proba")

        self.model = model.set_params(**model_params)
        self.bands = bands
        self.colors = self.get_color_permutations(bands)
        self._trained_models = dict()
        self._trained_models_scores = dict()

    def fit(self, X, y):
        #trained_models = dict()
        for each in self.colors:
            X_color = self.get_color_indices(X, each)
            clf = sklearn.base.clone(self.model)
            clf.fit(X_color, y)
            # Score serves as classifier
            score = int(100*clf.score(X_color, y))
            self._trained_models[each] = pickle.dumps(clf)
            self._trained_models_scores[each] = score
        return self

    def predict(self, X, y=None):
        predictions = []
        weights = []
        for color, model_string in self._trained_models.items():
        	clf = pickle.loads(model_string)
        	pred = clf.predict(X)
        	weight = self._trained_models_score[color]
        	weights.append(weight)
        	predictions.append(pred * weight)

        predictions = np.array(predictions) / np.sum(weights)
        y_pred = np.mean(predictions, axis=0)
        return y_pred

    def get_color_permutations(self, bands):
        # Color comboinations
        c_combo = list(itertools.combinations(bands,2))
        # Color-cut combinations
        cc_combo = list(itertools.combinations(c_combo,2))
        cc_labels = ["".join([cc[0][0],cc[0][1],cc[1][0],cc[1][1]]) for cc in cc_combo]

        return cc_labels

    def get_color_indices(self, X, color):
        # Make X from colors
        color_x = np.array(X[color[2]] - X[color[3]])
        color_y = np.array(X[color[0]] - X[color[1]])

        return np.vstack([color_x,color_y]).T
