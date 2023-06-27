# List of exercises done so far

[**Exercise 1**](exercise_1)

Goal of this exercise was visualization and clustering implementation. 

For visualization, a smoothed distribution of customer views upon an add was performed, 
  starting from a completely unstructured dataset. Result of this visualization is [here](exercise_1/heatmap_exercise_1_1.pdf) (with user look up on top [here](exercise_1/heatmap_wuser_looks_exercise_1_1.pdf)) and the corresponding implementation
[here](exercise_1/exercise1_1.py).

The clustering algorithm I implemented from scratch is the [Mean-Shift](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html) algorithm, using 
[scipy's KDTree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html) implementation for speedup boost for neighbor lookup. The comparison of my implementation
with the standard result of Mean-Shift on the same custom dataset I used is avaialble [here](exercise_1/clusters_exercise_1_2.pdf), while the main code is [here](exercise_1/exercise1_2.py).

[**Exercise 2**](exercise_2)

Here I have played around with the [Stree View House Number dataset](http://ufldl.stanford.edu/housenumbers/), extracting single digits from all the images and making a small classifier using simple CNN model.
The relevant [preprocessing](exercise_2/SVHN_preprocess.ipynb) and [google-colab training](exercise_2/SVHN_googlecolab_training.ipynb) notebooks are also available, alongside the library versions
I used in the corresponding [requirements.txt](exercise_2/requirements.txt) and the corresponding set of [utility scripts](exercise_2/lib/) I created.
