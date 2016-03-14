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

global TRAINING
TRAINING = 'ugrizTraining.fits'

def test_classifiers(N=250):
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

    # Select N (default=250) random objects from training file.
    y = training["iscwdm"]
    sss = StratifiedShuffleSplit(y, n_iter=1, test_size=N, train_size=N)
    trn_ind, tst_ind = tuple(sss)[0]

    # Make sure train/test are the same across all colors and classifiers
    X_train = training.iloc[trn_ind]
    y_train = training["iscwdm"].iloc[trn_ind]

    # Create dictionary to store fits
    fits = {}

    # Train a model for each color_cut_combination for each classifier being used.
    for name,clf in zip(classifier_names,classifiers):
        # Initialize the classifier
        # Train the classifier on each color cut
        cwdm = cwdmModel(model=clf, bands=filters)
        cwdm_trained = cwdm.fit(X_train,y_train)

        # Save model to dictionary
        fits[name] = cwdm_trained

    # Pickle dump
    the_time = str(time.time())
    filename = "".join([localpath,"data/","trained_models-", the_time, ".pkl"])
    print("\nINFO: Writing results to %s\n" % filename)
    with open(filename, "w") as f:
        pickle.dump(fits, f)

def plot_all_classifiers(filename=None):
    rcParams.update({'figure.autolayout': True})

    if filename:
        filename = localpath+'data/'+filename
    else:
        filename = localpath+'data/trained_models-1457903689.53.pkl'

    # Load file
    pkl_file = open(filename, 'rb')
    trained_models = pickle.load(pkl_file)

    #####################
    # Set up all the data
    # Grab classifiers names and add "data" at the front.
    classifier_names = np.append("data",trained_models.keys())
    # Load training data -- set up as pandas data frame.
    training = fileprep(TRAINING)
    # Select testing sample from the training data.
    y = training["iscwdm"]
    sss = StratifiedShuffleSplit(y, n_iter=1, test_size=150)
    trn_ind, tst_ind = tuple(sss)[0]
    # Make sure train/test are the same across all colors and classifiers
    X_test = training.iloc[tst_ind]
    y = training["iscwdm"].iloc[tst_ind]

    # Get the colors from one of the classifiers
    colors = trained_models[classifier_names[1]].colors

    #
    # Setting up plotting variables
    rows_per_page = 4
    ncols = len(classifier_names)

    ########################
    # Set up plotting region
    # Colors
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    # Open pdf
    pdf = PdfPages(localpath+"plots/all_ugriz_classifiers.pdf")
    # Set the first first figure. Will need to close it and re-initialize once
    # we plot rows_per_page
    figure = plt.figure(figsize=(ncols*3,rows_per_page*3))
    i, j = 1, 1 # plot_num

    for each in colors:
        # Get X for the specified color
        color_x = np.array(X_test[each[2]] - X_test[each[3]])
        color_y = np.array(X_test[each[0]] - X_test[each[1]])
        X = np.vstack([color_x,color_y]).T

        # Set-up the mesh grid
        x_min, x_max = X[:, 0].min() - .4, X[:, 0].max() + .4
        y_min, y_max = X[:, 1].min() - .4, X[:, 1].max() + .4
        h = 0.02 #mesh grid step size

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))


        for name in classifier_names:
            # Just plot the data first
            print i, each, name
            ax = plt.subplot(rows_per_page, ncols, i)

            if name == "data":
                ax.set_xlabel("".join([each[2],"-",each[3]]),fontsize=16)
                ax.set_ylabel("".join([each[0],"-",each[1]]),fontsize=16)

            else:
                model = pickle.loads(trained_models[name]._trained_models[each])

                if hasattr(model, "decision_function"):
                    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
                else:
                    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

                # Plot the decision boundary. For that, we will assign a color to each
                # point in the mesh [x_min, m_max]x[y_min, y_max].
                Z = Z.reshape(xx.shape)
                ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

                # Get score and label plot
                score = int(100*model.score(X, y))
                ax.text(xx.max() - .1, yy.min() + .1, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')

            ax.scatter(X[:, 0], X[:, 1], s=30, c=y, cmap=cm_bright, alpha=0.6)
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())

            # If the first row of the plot, print the column labels
            if j == 1: ax.set_title(name)

            # If hit maximum pages to plot, save figure and start a new one (page)
            if i == int(rows_per_page*ncols):
                # Make room for labels
                # Save the page
                pdf.savefig(figure)
                # Close the figure to save memory
                plt.close()
                # Re-initialize the figure
                figure = plt.figure(figsize=(ncols*3,rows_per_page*3))
                # Reset plot number and classifier title flag
                i, j = 1, 1
            else: i+=1

        # Done with row of colors
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
