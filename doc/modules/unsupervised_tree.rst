.. _unsupervised_tree:

===========================
Unsupervised Decision Trees
===========================

.. currentmodule:: sklearn.tree

In unsupervised learning, the goal is to identify patterns
or structure in data without using labeled examples. Clustering is a common
unsupervised learning technique that groups similar examples together
based on their features. Unsupervised tree models are an adaptive way of generating
clusters of samples. For information on supervised tree models, see :ref:`supervised_tree`

In this guide, we overview the :ref:`unsup_criterion` used for splitting unsupervised trees,
and methods for evaluating the quality of the tree model in :ref:`unsup_evaluation`.

.. _unsup_criterion:

Unsupervised Criterion
----------------------

Unsupervised tree models use a variety of criteria to split nodes.

Two-Means
~~~~~~~~~

The two means split finds the cutpoint that minimizes the one-dimensional
2-means objective, which is finding the cutoff point where the total variance
from cluster 1 and cluster 2 are minimal.

.. math::
  \min_s \sum_{i=1}^s (x_i - \hat{\mu}_1)^2 + \sum_{i=s+1}^N (x_i - \hat{\mu}_2)^2

where x is a N-dimensional feature vector, N is the number of sample_indices and
the \mu terms are the estimated means of each cluster 1 and 2.

Fast-BIC
~~~~~~~~

The Bayesian Information Criterion (BIC) is a popular model seleciton
criteria that is based on the log likelihood of the model given data.
Fast-BIC :footcite:`Meghana2019_geodesicrf` is a method that combines the speed of the
:class:`sklearn.cluster.KMeans` clustering method with the model flexibility
of Mclust-BIC. It sorts data for each feature and tries all possible splits to
assign data points to one of two Gaussian distributions based on their position
relative to the split.
The parameters for each cluster are estimated using maximum likelihood
estimation (MLE).The method performs hard clustering rather than soft
clustering like in GMM, resulting in a simpler calculation of the likelihood.

.. math::

  \hat{L} = \sum_{n=1}^s[\log\hat{\pi}_1+\log{\mathcal{N}(x_n;\hat{\mu}_1,\hat{\sigma}_1^2)}]
  + \sum_{n=s+1}^N[\log\hat{\pi}_2+\log{\mathcal{N}(x_n;\hat{\mu}_2,\hat{\sigma}_2^2)}]

where the prior, mean, and variance are defined as follows, respectively:

.. math::

  \hat{\pi} = \frac{s}{N},\quad\quad
  \hat{\mu} = \frac{1}{s}\sum_{n\le s}{x_n},\quad\quad
  \hat{\sigma}^2 = \frac{1}{s}\sum_{n\le s}{||x_n-\hat{\mu_j}||^2}

.. _unsup_evaluation:

Evaluating Unsupervised Trees
-----------------------------

In clustering settings, there may be no natural
notion of “true” class-labels, thus the efficacy of the clustering scheme is
often measured based on distance based metrics such as :func:`sklearn.metrics.adjusted_rand_score`.

.. topic:: References

 .. footbibliography::