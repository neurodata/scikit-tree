.. _oblique_forests:

Oblique Random Forests
----------------------

In oblique random forests (see :class:`~sktree.ObliqueRandomForestClassifier` and
`ObliqueRandomForestRegressor` classes), each tree in the ensemble is built
from a sample drawn with replacement (i.e., a bootstrap sample) from the
training set. The oblique random forest is the same as that of a random forest,
except in how the splits are computed in each tree.

Similar to how random forests achieve a reduced variance by combining diverse trees,
sometimes at the cost of a slight increase in bias, oblique random forests aim to do the same.
They are motivated to construct even more diverse trees, thereby improving model generalization.
In practice the variance reduction is often significant hence yielding an overall better model.

In contrast to the original publication :footcite:`breiman2001random`, the scikit-learn
implementation allows the user to control the number of features to combine in computing
candidate splits. This is done via the ``feature_combinations`` parameter. For
more information and intuition, see
:ref:`documentation on oblique decision trees <oblique_trees>`.

.. topic:: Examples:

 * :ref:`sphx_glr_auto_examples_plot_oblique_random_forest.py`
 * :ref:`sphx_glr_auto_examples_plot_oblique_axis_aligned_forests_sparse_parity.py`

.. topic:: References

 .. footbibliography::

.. _oblique_forest_feature_importance:

Feature importance evaluation
-----------------------------

The relative rank (i.e. depth) of a feature used as a decision node in a
tree can be used to assess the relative importance of that feature with
respect to the predictability of the target variable. Features used at
the top of the tree contribute to the final prediction decision of a
larger fraction of the input samples. The **expected fraction of the
samples** they contribute to can thus be used as an estimate of the
**relative importance of the features**. In scikit-tree, the fraction of
samples a feature contributes to is combined with the decrease in impurity
from splitting them to create a normalized estimate of the predictive power
of that feature. This is essentially exactly the same it is done in scikit-learn.

By **averaging** the estimates of predictive ability over several randomized
trees one can **reduce the variance** of such an estimate and use it
for feature selection. This is known as the mean decrease in impurity, or MDI.
Refer to [L2014]_ for more information on MDI and feature importance
evaluation with Random Forests. We implement the approach taken in :footcite:`Li2023manifold`
and :footcite:`TomitaSPORF2020`.

.. warning::

  The impurity-based feature importances computed on tree-based models suffer
  from two flaws that can lead to misleading conclusions. First they are
  computed on statistics derived from the training dataset and therefore **do
  not necessarily inform us on which features are most important to make good
  predictions on held-out dataset**. Secondly, **they favor high cardinality
  features**, that is features with many unique values.
  :ref:`sklearn:permutation_importance` is an alternative to impurity-based feature
  importance that does not suffer from these flaws. These two methods of
  obtaining feature importance are explored in:
  :ref:`sklearn:sphx_glr_auto_examples_inspection_plot_permutation_importance.py`.

In practice those estimates are stored as an attribute named
``feature_importances_`` on the fitted model. This is an array with shape
``(n_features,)`` whose values are positive and sum to 1.0. The higher
the value, the more important is the contribution of the matching feature
to the prediction function.

.. topic:: References

 .. footbibliography::

 .. [L2014] Louppe, G. :arxiv:`"Understanding Random Forests: From Theory to
    Practice" <1407.7502>`,
    PhD Thesis, U. of Liege, 2014.

