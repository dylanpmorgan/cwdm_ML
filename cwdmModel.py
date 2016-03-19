import os, time, sys, pdb
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle

import itertools
import sklearn.base
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

from matplotlib.colors import Normalize
from matplotlib import pyplot as plt

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
        self._is_trained = False
        self._best_params = dict()
        self._optimized_models = dict()
        self._optimized_models_scores = dict()
        self._is_optimized = False

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
        for each in self.colors:
            X_color_train = self.get_color_indices(X_train, each)
            X_color_test = self.get_color_indices(X_test, each)
            clf = sklearn.base.clone(self.model)
            clf.fit(X_color_train, y_train)
            # Score serves as classifier metric
            score = int(100*clf.score(X_color_test, y_test))
            self._trained_models[each] = pickle.dumps(clf)
            self._trained_models_scores[each] = score

        self._is_trained = True

        weighted_predictions = self.get_weighted_predictions(X_test)

        y_predictor = LogisticRegression()
        y_predictor.fit(weighted_predictions, y_test)
        self._y_predictor =  y_predictor

        return self

    def get_weighted_predictions(self, X):

        if self._is_trained:
            model_items = self._trained_models.items()
            model_weights = self._trained_models_scores
        elif self._is_optimized:
            model_items = self._optimized_models.items()
            model_weights = self._optimized_models_scores

        predictions = []
        weights = []
        for color, model_string in model_items:
            X_color = self.get_color_indices(X, color)

            clf = pickle.loads(model_string)
            pred = clf.predict(X_color)

            weight = model_weights[color]
            weights.append(weight)

            predictions.append(pred * weight)

        #predictions = np.array(predictions) / np.sum(weights)
        weighted_predictions = np.mean(predictions, axis=0)

        return weighted_predictions.reshape(-1,1)

    def predict(self, X, y=None):
        weighted_predictions = self.get_weighted_predictions(X)
        y_pred = self._y_predictor.predict(weighted_predictions)

        return y_pred

    def optimize_fit(self, X, y, param_grid, n_iter=100, check=False):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=100)
        for each in self.colors:
            X_color_train = self.get_color_indices(X_train, each)
            X_color_test = self.get_color_indices(X_test, each)

            clf = sklearn.base.clone(self.model)
            opt = GridSearchCV(clf, param_grid=param_grid,
                               verbose=True, cv=3)
            opt.fit(X_color_train, y_train)

            print("The best parameters are %s with a score of %0.2f"
                    % (opt.best_params_, opt.best_score_))

            if check:
                plot_validation(opt)

            clf.set_params(opt.best_params_)
            clf.fit(X_color_train, y_train)

            score = int(100*clf.score(X_color_test, y_test))

            self._best_params[each] = opt.best_params_
            self._optimized_models[each] = pickle.dumps(clf)
            self._optimized_scores[each] = score

        self._is_optimized = True
        weighted_predictions = self.get_weighted_predictions(X_test)

        y_predictor = LogisticRegression()
        y_predictor.fit(weighted_predictions, y_test)
        self._y_predictor =  y_predictor

        return self

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

    def plot_validation(self, opt):
        # Plot the Validation accuarcy of model optimizer

        scores = [x[1] for x in opt.grid_scores_]
        scores = np.array(scores).reshape(len(param_grid["C"]), len(param_grid["gamma"]))

        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
        plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
        plt.xlabel('gamma')
        plt.ylabel('C')
        plt.colorbar()
        plt.xticks(np.arange(len(param_grid["gamma"])), param_grid["gamma"], rotation=45)
        plt.yticks(np.arange(len(param_grid["C"])), param_grid["C"])
        plt.title('Validation accuracy')
        plt.show()
