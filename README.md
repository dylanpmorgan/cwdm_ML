## cwdm_ML

## Purpose

This is a code designed to use supervised ensemble classification algorithms
to identify a special subset of stars in color-color space using the filter
bandpasses (ugriz) from the sloan digital sky survey.

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
supervised ensemble classification algorithms from sklearn in more accurately
identifying WD+dM pairs.

![uz_rz_cuts](https://cloud.githubusercontent.com/assets/10521443/13748496/7424b6ac-e9d3-11e5-9c82-234316fceebc.png)

## Progress

Right now I am testing the different ensemble classification algorithms as shown below.

![testing_classifiers](https://cloud.githubusercontent.com/assets/10521443/13748495/741ccfd2-e9d3-11e5-858b-1c68aae66583.png)
[all_ugriz_classifiers.pdf](https://github.com/dylanpmorgan/cwdm_ML/files/172182/all_ugriz_classifiers.pdf)

The next steps are too optimize the fitting parameters for each classifier and
decide which classifier is the best performing. Then, we will apply all the trained models
to predict the probability of each star being a WD+dM.
