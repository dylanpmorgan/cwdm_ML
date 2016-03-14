from cwdmModel import cwdmModel

import os, time, sys, pdb

try:
    import cPickle as pickle
except:
    import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
import itertools
import pyfits
import pandas as pd
from astropy.table import Table
import collections

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import ShuffleSplit, StratifiedShuffleSplit

global localpath
localpath = '/Users/dpmorg/gdrive/research/cwdm_ML/'

def test_classifiers():
    TRAINING = 'ugrizTraining.fits'

    # Load training data -- set up as pandas data frame.
    training = fileprep(TRAINING)

    # Classifiers being used.
    classifier_names = ["Nearest Neighbors","RBF SVM","Random Forest"]
    classifiers = [
        KNeighborsClassifier(10),
        SVC(gamma=3, C=1),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)]

    # Setting up the color-color combinations
    filters = 'ugriz'
    color_combinations = list(itertools.combinations(filters,2))
    color_cut_combinations = list(itertools.combinations(color_combinations,2))
    # Unpack color labels
    color_labels = ["".join([cc[0][0],cc[0][1],cc[1][0],cc[1][1]]) for cc in color_cut_combinations]

    # Training and test size
    N = 250
    # Select randomly from training and testing
    y = training["iscwdm"]
    sss = StratifiedShuffleSplit(y, n_iter=1, test_size=N, train_size=N)
    trn_ind, tst_ind = tuple(sss)[0]

    # Make sure train/test are the same across all colors and classifiers
    train_smp = training.iloc[trn_ind]
    test_smp = training.iloc[tst_ind]

    fits = pd.DataFrame(index=color_labels, columns=classifier_names)
    scores = pd.DataFrame(index=color_labels, columns=classifier_names)

    # Train a model for each color_cut_combination for each classifier being used.
    for name,clf in zip(classifier_names,classifiers):
        # Initialize the classifier
        # Train the classifier on each color cut
        for label, color_cut in zip(color_labels,color_cut_combinations):
            # Pull out the colors from the training
            trn_x = np.array(train_smp[color_cut[1][0]]-
                             train_smp[color_cut[1][1]])
            trn_y = np.array(train_smp[color_cut[0][0]]-
                             train_smp[color_cut[0][1]])

            X_trn, y_trn = np.vstack([trn_x,trn_y]).T, train_smp['iscwdm']

            # Pull out the colors from the testing
            tst_x = np.array(test_smp[color_cut[1][0]]-
                             test_smp[color_cut[1][1]])
            tst_y = np.array(test_smp[color_cut[0][0]]-
                             test_smp[color_cut[0][1]])

            X_tst, y_tst = np.vstack([tst_x,tst_y]).T, test_smp['iscwdm']

            # Initiate the model
            cwdm = cwdmModel(X_trn, y_trn, X_tst, y_tst, model=clf)

            # Train the classifier and get the score
            cwdm.fit(X_trn, y_trn)
            score = cwdm.score(X_tst, y_tst)

            fits.loc[label,name] = cwdm
            scores.loc[label,name] = score

    # Combine fits and scores
    panel = pd.Panel({"fits" : fits, "scores" : scores})

    # Dump everything into a file
    the_time = str(time.time())
    filename = "".join([localpath,"data/","trained_models-", the_time, ".pkl"])
    print("\nINFO: Writing results to %s\n" % filename)
    with open(filename, "w") as f:
        pickle.dump(panel, f)

def plot_all_classifiers(filename=None):
    rcParams.update({'figure.autolayout': True})

    if filename:
        filename = localpath+'data/'+filename
    else:
        filename = localpath+'data/trained_models-1457903689.53.pkl'

    # Load file
    pkl_file = open(filename, 'rb')
    trained_models = pickle.load(pkl_file)

    # Grab row and column names
    color_cut_combinations = trained_models['fits'].index
    classifiers = trained_models['fits'].columns

    # Setting up plotting variables
    ncols = len(classifiers)+1
    nrows = len(color_cut_combinations)
    rows_per_page = 4
    npages = int(round(nrows/rows_per_page))

    # Open pdf
    pdf = PdfPages(localpath+"plots/all_ugriz_classifiers.pdf")

    # Set the first first figure. Will need to close it and re-initialize once
    # we plot rows_per_page
    figure = plt.figure(figsize=((len(classifiers)+1)*3,rows_per_page*3))
    i, j = 1, 1 # plot_num

    for color_cut in color_cut_combinations:
        # Just plot the data set first
        data = trained_models['fits'].loc[color_cut,classifiers[0]]

        X_train = data.X_train
        y_train = data.y_train
        X_test = data.X_test
        y_test = data.y_test

        # Set-up mesh grid
        x_min, x_max = X_train[:, 0].min() - .4, X_train[:, 0].max() + .4
        y_min, y_max = X_train[:, 1].min() - .4, X_train[:, 1].max() + .4
        h = 0.02 #mesh grid step size

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        print i, color_cut
        ax = plt.subplot(rows_per_page, len(classifiers) + 1, i)
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], s=20, c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], s=40, c=y_test, cmap=cm_bright, alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel("".join([color_cut[2],"-",color_cut[3]]),fontsize=16)
        ax.set_ylabel("".join([color_cut[0],"-",color_cut[1]]),fontsize=16)

        # Increment plot number
        i += 1

        for name in classifiers:
            print i, color_cut, name
            ax = plt.subplot(rows_per_page, len(classifiers) + 1, i)

            # Get the model and score
            data = trained_models['fits'].loc[color_cut,name]
            #fit = pickle.loads(data.trained)
            score = trained_models['scores'].loc[color_cut,name]

            # Grab the data points
            # -- Should the same for every color cut, so this is kind of
            #    repetitive. May change in the future
            X_train = data.X_train
            y_train = data.y_train
            X_test = data.X_test
            y_test = data.y_test

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, m_max]x[y_min, y_max].
            if hasattr(data.clf, "decision_function"):
                Z = data.clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = data.clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

            # Plot also the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], s=20, c=y_train, cmap=cm_bright)
            # and testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], s=40, c=y_test, cmap=cm_bright,
                       alpha=0.6)

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.text(xx.max() - .1, yy.min() + .1, ('%.2f' % score).lstrip('0'),
            size=15, horizontalalignment='right')
            plt.gcf().subplots_adjust(bottom=0.10,left=0.10,right=0.90,top=0.90)

            # If on the first row of the page, print titles
            if j == 1: ax.set_title(name)

            if i == int(rows_per_page*ncols):
                # Make room for labels
                # Save the page
                pdf.savefig(figure)
                # Close the figure to save memory
                plt.close()
                # Re-initialize the figure
                figure = plt.figure(figsize=((len(classifiers)+1)*3,rows_per_page*3))
                # Reset plot number and classifier title flag
                i, j = 1, 0
            else: i += 1

        # Done with first row, increment so it no longer labels plot
        j += 1

    # Close pdf
    pdf.close()

def fileprep(filename):
    # Getting rid of astropy.table format. Annoying to use.
    t = Table.read(localpath+'data/'+filename)

    data = collections.OrderedDict()
    for col in t.colnames:
        data[col.lower()] = np.array(t[col]).byteswap().newbyteorder()

    df = pd.DataFrame.from_dict(data)

    # Quality cuts on the data
    quality_cut = np.where((df['u_e'] <= 3.*df['u_e'].median()) &
                           (df['g_e'] <= 3.*df['g_e'].median()) &
                           (df['r_e'] <= 3.*df['r_e'].median()) &
                           (df['i_e'] <= 3.*df['i_e'].median()) &
                           (df['z_e'] <= 3.*df['z_e'].median()) &
                           (df['u'] > 0.) & (df['u'] <= 22.0) &
                           (df['g'] > 0.) & (df['g'] <= 21.5) &
                           (df['r'] > 0.) & (df['r'] <= 21.5) &
                           (df['i'] > 0.) & (df['i'] <= 21.5) &
                           (df['z'] > 0.) & (df['z'] <= 21.5))[0]

    df = df.iloc[quality_cut]

    # Color cuts on the data
    color_cut = np.where((df['u'] - df['g'] >= -1) &
                         (df['u'] - df['g'] <= 2) &
                         (df['g'] - df['r'] > -0.6) &
                         (df['g'] - df['r'] < 1.4))[0]

    df = df.iloc[color_cut]

    return df
