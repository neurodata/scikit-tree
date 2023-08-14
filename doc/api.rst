.. _api_documentation:

=================
API Documentation
=================

:py:mod:`sktree`:

.. automodule:: sktree
   :no-members:
   :no-inherited-members:

Scikit-learn Tree Estimators
----------------------------
We provide a drop-in replacement for the scikit-learn tree estimators
with **experimental** features that we have developed. These estimators
are still compatible with the scikit-learn API. These estimators all have
the capability of binning features, which theoretically will improve runtime
significantly for high-dimensional and high-sample size data.

Use at your own risk! We have not tested these estimators extensively, compared
to the scikit-learn estimators.

.. automodule:: sktree._lib.sklearn.ensemble
   :members:
   :show-inheritance:

.. currentmodule:: sktree
.. autosummary::
   :toctree: generated/
   
   RandomForestClassifier
   RandomForestRegressor
   ExtraTreesClassifier
   ExtraTreesRegressor

Supervised
----------
Decision-tree models are traditionally implemented with axis-aligned splits and
storing the mean outcome (i.e. label) vote in the leaf nodes. However, more
exotic splits are possible, called "oblique" splits, which are some function
of multiple feature columns to create a "new feature value" to split on.

This can take the form of a random (sparse) linear combination of feature columns,
or even take advantage of the structure in the data (e.g. if it is an image) to
sample feature indices in a manifold-aware fashion. This class of models generalizes
the splitting function in the trees, while everything else is consistent with
how scikit-learn builds trees.

.. currentmodule:: sktree
.. autosummary::
   :toctree: generated/

   ObliqueRandomForestClassifier
   ObliqueRandomForestRegressor
   PatchObliqueRandomForestClassifier
   PatchObliqueRandomForestRegressor
   HonestForestClassifier

.. currentmodule:: sktree.tree
.. autosummary::
   :toctree: generated/

   ObliqueDecisionTreeClassifier
   ObliqueDecisionTreeRegressor
   PatchObliqueDecisionTreeClassifier
   PatchObliqueDecisionTreeRegressor
   HonestTreeClassifier

Unsupervised
------------
Decision-tree models are traditionally used for classification and regression.
However, they are also powerful non-parametric embedding and clustering models.
The :class:`~sklearn.ensemble.RandomTreesEmbedding` is an example of unsupervised
tree model. We implement other state-of-the-art models that explicitly split based
on unsupervised criterion such as variance and BIC.

.. currentmodule:: sktree
.. autosummary::
   :toctree: generated/

   UnsupervisedRandomForest
   UnsupervisedObliqueRandomForest

The trees that comprise those forests are also available as standalone classes.

.. autosummary::
   :toctree: generated/

   tree.UnsupervisedDecisionTree
   tree.UnsupervisedObliqueDecisionTree


Distance Metrics
----------------
Trees inherently produce a "distance-like" metric. We provide an API for
extracting pairwise distances from the trees that include a correction
that turns the "tree-distance" into a proper distance metric.

.. currentmodule:: sktree.tree
.. autosummary::
   :toctree: generated/

   compute_forest_similarity_matrix

In addition to providing a distance metric based on leaves, tree-models
provide a natural way to compute neighbors based on the splits. We provide
an API for extracting the nearest neighbors from a tree-model. This provides
an API-like interface similar to :class:`~sklearn.neighbors.NearestNeighbors`.

.. currentmodule:: sktree
.. autosummary::
   :toctree: generated/

   NearestNeighborsMetaEstimator


Experimental Functionality
--------------------------
We also include experimental functionality that is works in progress.

.. currentmodule:: sktree.experimental
.. autosummary::
   :toctree: generated/

   mutual_info_ksg

We also include functions that help simulate and evaluate mutual information (MI)
and conditional mutual information (CMI) estimators. Specifically, functions that
help simulate multivariate gaussian data and compute the analytical solutions
for the entropy, MI and CMI of the Gaussian distributions.

.. currentmodule:: sktree.experimental.simulate
.. autosummary::
   :toctree: generated/

   simulate_multivariate_gaussian
   simulate_helix
   simulate_sphere

.. currentmodule:: sktree.experimental.mutual_info
.. autosummary::
   :toctree: generated/

   mi_gaussian
   cmi_gaussian
   entropy_gaussian

.. currentmodule:: sktree.experimental.monte_carlo
.. autosummary::
   :toctree: generated/

   conditional_resample
