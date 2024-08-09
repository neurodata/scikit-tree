import copy

import numpy as np
from scipy.sparse import issparse

from .._lib.sklearn.tree._criterion import BaseCriterion
from .._lib.sklearn.tree._tree import BestFirstTreeBuilder, DepthFirstTreeBuilder
from ._classes import CRITERIA_CLF, PATCH_DENSE_SPLITTERS, PatchObliqueDecisionTreeClassifier
from ._oblique_tree import ObliqueTree


def gaussian_kernel(shape, sigma=1.0, mu=0.0):
    """N-dimensional gaussian kernel for the given shape.

    See: https://gist.github.com/liob/e784775e882b83749cb3bbcef480576e
    """
    m = np.meshgrid(*[np.linspace(-1, 1, s) for s in shape])
    d = np.sqrt(np.sum([x * x for x in m], axis=0))
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma**2)))
    return g / np.sum(g)


class KernelDecisionTreeClassifier(PatchObliqueDecisionTreeClassifier):
    """Oblique decision tree classifier over data patches combined with Gaussian kernels.

    Parameters
    ----------
    criterion : {"gini", "entropy"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.
    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
        `ceil(min_samples_split * n_samples)` are the minimum
        number of samples for each split.
    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
        `ceil(min_samples_leaf * n_samples)` are the minimum
        number of samples for each node.
    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
    max_features : int, float or {"auto", "sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split:
            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
            `int(max_features * n_features)` features are considered at each
            split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
        Note: Compared to axis-aligned Random Forests, one can set
        max_features to a number greater then ``n_features``.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split, even if ``splitter`` is set to
        ``"best"``. When ``max_features < n_features``, the algorithm will
        select ``max_features`` at random at each split before finding the best
        split among them. But the best found split may vary across different
        runs, even if ``max_features=n_features``. That is the case, if the
        improvement of the criterion is identical for several splits and one
        split has to be selected at random. To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.
        See :term:`Glossary <random_state>` for details.
    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        The weighted impurity decrease equation is the following::
            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)
        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.
        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.
    class_weight : dict, list of dict or "balanced", default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If None, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.
        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``
        For multi-output, the weights of each column of y will be multiplied.
        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.
    min_patch_dim : array-like, optional
        The minimum dimensions of a patch, by default 1 along all dimensions.
    max_patch_dim : array-like, optional
        The maximum dimensions of a patch, by default 1 along all dimensions.
    dim_contiguous : array-like of bool, optional
        Whether or not each patch is sampled contiguously along this dimension.
    data_dims : array-like, optional
        The presumed dimensions of the un-vectorized feature vector, by default
        will be a 1D vector with (1, n_features) shape.
    boundary : optional, str {'wrap'}
        The boundary condition to use when sampling patches, by default None.
        'wrap' corresponds to the boundary condition as is in numpy and scipy.
    feature_weight : array-like of shape (n_samples,n_features,), default=None
        Feature weights. If None, then features are equally weighted as is.
        If provided, then the feature weights are used to weight the
        patches that are generated. The feature weights are used
        as follows: for every patch that is sampled, the feature weights over
        the entire patch is summed and normalizes the patch.
    kernel : str {'gaussian', 'uniform'}, default='gaussian'
        The kernel to use.
    n_kernels : int, optional
        The number of different kernels to generate. This number should be very high
        as this generates kernels
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,) or list of ndarray
        The classes labels (single output problem),
        or a list of arrays of class labels (multi-output problem).
    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance [4]_.
        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.
    max_features_ : int
        The inferred value of max_features.
    n_classes_ : int or list of int
        The number of classes (for single output problems),
        or a list containing the number of classes for each
        output (for multi-output problems).
    n_features_in_ : int
        Number of features seen during :term:`fit`.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    n_outputs_ : int
        The number of outputs when ``fit`` is performed.
    tree_ : Tree instance
        The underlying Tree object. Please refer to
        ``help(sklearn.tree._tree.Tree)`` for
        attributes of Tree object.
    min_patch_dims_ : array-like
        The minimum dimensions of a patch.
    max_patch_dims_ : array-like
        The maximum dimensions of a patch.
    data_dims_ : array-like
        The presumed dimensions of the un-vectorized feature vector.
    kernel_arr_ : list of length (n_nodes,) of array-like of shape (patch_dims,)
        The kernel array that is applied to the patches for this tree.
        The order is in the same order in which the tree traverses the nodes.
    kernel_params_ : list of length (n_nodes,)
        The parameters of the kernel that is applied to the patches for this tree.
        The order is in the same order in which the tree traverses the nodes.
    kernel_library_ : array-like of shape (n_kernels,), optional
        The library of kernels that was chosen from the patches for this tree.
        Only stored if ``store_kernel_library`` is set to True.
    kernel_library_params_ : list of length (n_nodes,), optional
        The parameters of the kernels that was chosen from the patches for this tree.
        The order is in the same order in which the tree traverses the nodes.
        Only stored if ``store_kernel_library`` is set to True.
    References
    ----------
    .. footbibliography::
    """

    _parameter_constraints: dict = {**PatchObliqueDecisionTreeClassifier._parameter_constraints}
    _parameter_constraints.update(
        {
            "kernel": ["str"],
            "n_kernels": [int, None],
            "store_kernel_library": [bool],
        }
    )

    def __init__(
        self,
        *,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        class_weight=None,
        min_patch_dims=None,
        max_patch_dims=None,
        dim_contiguous=None,
        data_dims=None,
        boundary=None,
        feature_weight=None,
        kernel="gaussian",
        n_kernels=None,
        store_kernel_library=False,
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
        )

        self.min_patch_dims = min_patch_dims
        self.max_patch_dims = max_patch_dims
        self.dim_contiguous = dim_contiguous
        self.data_dims = data_dims
        self.boundary = boundary
        self.feature_weight = feature_weight

        self.kernel = kernel
        self.n_kernel = n_kernels
        self.store_kernel_library = store_kernel_library

    def _build_tree(
        self,
        X,
        y,
        sample_weight,
        min_samples_leaf,
        min_weight_leaf,
        max_leaf_nodes,
        min_samples_split,
        max_depth,
        random_state,
    ):
        """Build the actual tree.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels) as integers or strings.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.
        min_samples_leaf : int or float
            The minimum number of samples required to be at a leaf node.
        min_weight_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights.
        max_leaf_nodes : int, default=None
            Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        min_samples_split : int or float, default=2
            The minimum number of samples required to split an internal node:
        max_depth : int, default=None
            The maximum depth of the tree. If None, then nodes are expanded until
            all leaves are pure or until all leaves contain less than
            min_samples_split samples.
        random_state : int, RandomState instance or None, default=None
            Controls the randomness of the estimator.
        """
        # Build tree
        criterion = self.criterion
        if not isinstance(criterion, BaseCriterion):
            criterion = CRITERIA_CLF[self.criterion](self.n_outputs_, self.n_classes_)
        else:
            # Make a deepcopy in case the criterion has mutable attributes that
            # might be shared and modified concurrently during parallel fitting
            criterion = copy.deepcopy(criterion)

        splitter = self.splitter
        if issparse(X):
            raise ValueError(
                "Sparse input is not supported for oblique trees. "
                "Please convert your data to a dense array."
            )
        else:
            PATCH_SPLITTERS = PATCH_DENSE_SPLITTERS

        # compute user-defined Kernel library before
        kernel_library, kernel_dims, kernel_params = self._sample_kernel_library(X, y)

        # Note: users cannot define a splitter themselves
        splitter = PATCH_SPLITTERS[self.splitter](
            criterion,
            self.max_features_,
            min_samples_leaf,
            min_weight_leaf,
            random_state,
            self.min_patch_dims_,
            self.max_patch_dims_,
            self.dim_contiguous_,
            self.data_dims_,
            self.boundary,
            self.feature_weight,
            kernel_library,
        )

        self.tree_ = ObliqueTree(self.n_features_in_, self.n_classes_, self.n_outputs_)

        # Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
        if max_leaf_nodes < 0:
            builder = DepthFirstTreeBuilder(
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                self.min_impurity_decrease,
            )
        else:
            builder = BestFirstTreeBuilder(
                splitter,
                min_samples_split,
                min_samples_leaf,
                min_weight_leaf,
                max_depth,
                max_leaf_nodes,
                self.min_impurity_decrease,
            )

        builder.build(self.tree_, X, y, sample_weight)

        # Set some attributes based on what kernel indices were used in each tree
        # - construct a tree-nodes like array and set it as a Python attribute, so it is
        #   exposed to the Python interface
        # - use the Cython tree.feature array to store the index of the dictionary library
        #   that was used to split at each node
        kernel_idx_chosen = self.tree_.feature
        kernel_lib_chosen = kernel_library[kernel_idx_chosen]
        kernel_params_chosen = kernel_params[kernel_idx_chosen]
        self.kernel_arr_ = kernel_lib_chosen
        self.kernel_dims_ = kernel_dims[kernel_idx_chosen]
        self.kernel_params_ = kernel_params_chosen

        if self.store_kernel_library:
            self.kernel_library_ = kernel_library
            self.kernel_idx_chosen_ = kernel_idx_chosen

        if self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

    def _sample_kernel_library(self, X, y, sample_weight):
        """Samples the dictionary library that is sampled from in the random forest.
        A kernel can either be applied with the boundaries of the image in mind, such that
        the patch can be uniformly applied across all indices of rows of ``X_structured``. This
        is equivalent to passing in ``boundary = 'wrap'``. Alternatively, a kernel can be sampled,
        such that the patch is always contained within the boundary of the image. This is equivalent
        to passing in ``boundary = None``.
        Parameters
        ----------
        X : _type_
            _description_
        y : _type_
            _description_
        sample_weight : _type_
            _description_
        Returns
        -------
        kernel_library : array-like of shape (n_kernels, n_features)
            The dictionary that will be sampled from the random forest. Non-zero entries indicate
            where the kernel is applied. This is a n-dimensional kernel matrix that is vectorized
            and placed into the original ``X`` data dimensions of (n_dims,).
        kernel_dims : list of (n_kernels,) length
            The dimensions of each kernel that is sampled. For example, (n_dims_x, n_dims_y) in
            the first element indicates that the first kernel has a shape of (n_dims_x, n_dims_y).
            This can be used to then place where the top-left seed of each kernel is applied.
        kernel_params : list of (n_kernels,) length
            A list of dictionaries representing the parameters of each kernel. This is used to
            keep track of what kernels and parameterizations were valuable when used in the random
            forest.
        """
        raise NotImplementedError("This method should be implemented in a child class.")


class GaussianKernelDecisionTreeClassifier(KernelDecisionTreeClassifier):
    _parameter_constraints = {
        **KernelDecisionTreeClassifier._parameter_constraints,
        # "mu_bounds": ['array-like'],
        # "sigma_bounds": ['array-like'],
    }

    def __init__(
        self,
        *,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0,
        class_weight=None,
        min_patch_dims=None,
        max_patch_dims=None,
        dim_contiguous=None,
        data_dims=None,
        boundary=None,
        feature_weight=None,
        n_kernels=None,
        store_kernel_library=False,
        mu_bounds=(0, 1),
        sigma_bounds=(0, 1),
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            class_weight=class_weight,
            min_patch_dims=min_patch_dims,
            max_patch_dims=max_patch_dims,
            dim_contiguous=dim_contiguous,
            data_dims=data_dims,
            boundary=boundary,
            feature_weight=feature_weight,
            kernel="gaussian",
            n_kernels=n_kernels,
            store_kernel_library=store_kernel_library,
        )
        self.mu_bounds = mu_bounds
        self.sigma_bounds = sigma_bounds

    def _sample_kernel_library(self, X, y, sample_weight):
        """Samples a gaussian kernel library."""
        rng = np.random.default_rng(self.random_state)
        kernel_library = []
        kernel_params = []
        kernel_dims = []

        # Sample the kernel library
        ndim = len(self.data_dims_)
        for _ in range(self.n_kernels):
            patch_shape = []
            for idx in range(ndim):
                # Note: By constraining max patch height/width to be at least the min
                # patch height/width this ensures that the minimum value of
                # patch_height and patch_width is 1
                patch_dim = rng.randint(self.min_patch_dims_[idx], self.max_patch_dims_[idx] + 1)

                # sample the possible patch shape given input parameters
                if self.boundary == "wrap":
                    # add circular boundary conditions
                    delta_patch_dim = self.data_dims_[idx] + 2 * (patch_dim - 1)

                    # sample the top left index for this dimension
                    top_left_patch_seed = rng.randint(0, delta_patch_dim)

                    # resample the patch dimension due to padding
                    dim = top_left_patch_seed % delta_patch_dim

                    # resample the patch dimension due to padding
                    patch_dim = min(
                        patch_dim, min(dim + 1, self.data_dims_[idx] + patch_dim - dim - 1)
                    )

                patch_shape.append(patch_dim)

            patch_shape = np.array(patch_shape)

            # sample the sigma and mu parameters
            sigma = rng.uniform(low=self.mu_bounds[0], high=self.mu_bounds[1])
            mu = rng.uniform(low=self.sigma_bounds[0], high=self.sigma_bounds[1])

            kernel = gaussian_kernel(shape=patch_shape, sigma=sigma, mu=mu)

            kernel_dims.append(kernel.shape)
            kernel_library.append(kernel)
            kernel_params.append({"shape": patch_shape, "sigma": sigma, "mu": mu})

        return kernel_library, kernel_dims, kernel_params
