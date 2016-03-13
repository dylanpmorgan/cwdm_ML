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

![uz_rz_cuts](https://cloud.githubusercontent.com/assets/10521443/13731713/5ed55adc-e946-11e5-95af-d391d531e870.png)

## Progress

Right now I am testing the different ensemble classification algorithms as shown below.

![testing_classifiers](https://cloud.githubusercontent.com/assets/10521443/13731712/5ed35b06-e946-11e5-85a0-6fb35f794ee5.png)
[all_ugriz_classifiers.pdf](https://github.com/dylanpmorgan/cwdm_ML/files/171253/all_ugriz_classifiers.pdf)

The next steps are too optimize the fitting parameters for each classifier and
decide which classifier is the best performing. Then, we will apply all the trained models
to predict the probability of each star being a WD+dM.
