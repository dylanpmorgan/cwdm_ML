## cwdm_ML

## Purpose

This is a code designed to use supervised ensemble classification algorithms
to identify a special subset of stars in color-color space using the filter
bandpasses (ugriz) from the Sloan Digital Sky Survey (SDSS).

## Science motivation

Close, unresolved white dwarf + M dwarf (WD+dM) binary pairs are a special class of binary
stars that include a degenerate stellar remnant (white dwarf) and a low-mass star (M dwarf).
These systems are particularly useful because, given their unique colors (appearing
as a single star that's both very blue and very red), they are relatively easy to identify
from other astrophysical objects. Close binary star systems are otherwise very
difficult to identify. With their unique colors, astronomers are able to build
large statistical samples of WD+dM and use them as proxies for understanding
close binary systems in general.

Traditional methods of selecting WD+dM pairs rely on linear cuts in color-color
space, typically done by eye (shown below). With this code, I explore the merits of using
supervised ensemble classification algorithms from scikit-learn in more accurately
identifying WD+dM pairs.

![uz_rz_cuts](https://cloud.githubusercontent.com/assets/10521443/13748496/7424b6ac-e9d3-11e5-9c82-234316fceebc.png)

## Progress

Presently, I'm testing three different classifiers: Radial Basis Function using
Support Vector Machine (RBF SVM), Random Forest, and Nearest Neighbors.

![testing_classifiers](https://cloud.githubusercontent.com/assets/10521443/13748495/741ccfd2-e9d3-11e5-858b-1c68aae66583.png)
[all_ugriz_classifiers.pdf](https://github.com/dylanpmorgan/cwdm_ML/files/172182/all_ugriz_classifiers.pdf)

After training each classifier, I use the trained models for all of possible
color-color combinations and make a prediction on the object being a WD+dM pair or not.
The prediction is made by using a combined score-weighted prediction for each
object (ranging from 0-100). I then use logistic regression to optimize the
combined score-weighted prediction for best performance. The performance of each classifier
is shown below:

| Model name   | RBF SVM | Nearest Neighbors | Random Forest |
|--------------|---------|-------------------|---------------|
| Precision    | 87%     | 89%               | 87%           |
| Recall       | 84%     | 87%               | 85%           |
| WD+dMs found | 1543/1831 | 1601/1831 | 1563/1831 |
| WD+dMs misclassified | 240 | 193 | 228 |
| WD+dMs missed | 288 | 230 | 268 |

Precision: # predicted WD+dMs / # of predicted WD+dMs are real.
Recall: # of WD+dMs correctly predicted / total # of WD+dMs in the sample.

## Next Steps

The next steps are to optimize each classifier. To do this, I am performing an
exhaustive parameter grid search over each of the classifiers to determine which
parameters are most appropriate for the data. For now, I'm finding optimizing each
classifier for each color-color space, hopefully this ends up being redundant as
it is computationally intensive.
