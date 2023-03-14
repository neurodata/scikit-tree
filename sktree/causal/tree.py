from sklearn.base import BaseEstimator
from sklearn.ensemble._forest import BaseForest, ForestRegressor
from sklearn.tree import BaseDecisionTree, DecisionTreeClassifier, DecisionTreeRegressor


class CausalTree(BaseEstimator):
    def __init__(
        self,
        tree=None,
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
        ccp_alpha=0.0,
        honest_ratio=0.5,
        **tree_params,
    ) -> None:
        """A causal tree.

        A causal tree is a decision tree model that operates over not only
        ``(X, y)`` tuple of arrays, but also includes the treatment groups.
        It is an estimator model that has a graphical model of the form
        :math:`T \\rightarrow Y` with :math:`T \\leftarrow X \\rightarrow Y`.

        Parameters
        ----------
        tree : _type_, optional
            _description_, by default None
        criterion : str, optional
            _description_, by default "gini"
        splitter : str, optional
            _description_, by default "best"
        max_depth : _type_, optional
            _description_, by default None
        min_samples_split : int, optional
            _description_, by default 2
        min_samples_leaf : int, optional
            _description_, by default 1
        min_weight_fraction_leaf : float, optional
            _description_, by default 0.0
        max_features : _type_, optional
            _description_, by default None
        random_state : _type_, optional
            _description_, by default None
        max_leaf_nodes : _type_, optional
            _description_, by default None
        min_impurity_decrease : float, optional
            _description_, by default 0.0
        class_weight : _type_, optional
            _description_, by default None
        ccp_alpha : float, optional
            _description_, by default 0.0
        honest_ratio : float, optional
            _description_, by default 0.5

        Notes
        -----
        Compared to a normal decision tree model, which estimates the value :math:`E[Y | X]`,
        a causal decision tree splits the data in such a way to estimate the value
        :math:`E[Y | do(T), X]`. This is a causal effect, and thus requires causal assumptions.
        The core assumption is that of unconfoundedness, where the outcomes, ``Y``, and the
        treatments, ``T`` are not confounded by any other non-observed variables. That is ``X``
        captures all possible confounders. This is typically a strong assumption, but is realized
        in well-designed observational studies and randomized control trials, where the treatment
        assignment is randomized.
        """
        self.tree = tree
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.honest_ratio = honest_ratio
        self.tree_params = tree_params

    def fit(self, X, y, sample_weight=None, T=None):
        # initialize the tree estimator
        if self.tree is None:
            self.estimator_ = DecisionTreeClassifier(
                criterion=self.criterion,
                splitter=self.splitter,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_features=self.max_features,
                random_state=self.random_state,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                class_weight=self.class_weight,
                ccp_alpha=self.ccp_alpha,
                honest_ratio=1.0,
                **self.tree_params,
            )
        else:
            if not issubclass(self.tree, BaseDecisionTree):
                raise RuntimeError(
                    "The type of tree must be a subclass of sklearn.tree.BaseDecisionTree."
                )

            self.estimator_ = self.tree(
                criterion=self.criterion,
                splitter=self.splitter,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_features=self.max_features,
                random_state=self.random_state,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                class_weight=self.class_weight,
                ccp_alpha=self.ccp_alpha,
                honest_ratio=1.0,
                **self.tree_params,
            )

        # fit the decision tree using only the treatment groups
        self.estimator_.fit(X, y=T, sample_weight=sample_weight)

        # re-fit the tree leaves using the outcomes
        refit_leaves(self.estimator_, X, y)

    def apply(self, X):
        return self.estimator_.apply(X)

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def decision_path(self, X):
        return self.estimator_.decision_path(X)

    @property
    def feature_importances_(self):
        """Return the feature importances.

        The importance of a feature is computed as the (normalized) total
        reduction of the criterion brought by that feature.
        It is also known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            Normalized total reduction of criteria by feature
            (Gini importance).
        """
        return self.estimator_.feature_importances_


class ForestCausalRegressor(BaseForest):
    @abstractmethod
    def __init__(
        self,
        estimator,
        n_estimators=100,
        *,
        estimator_params=tuple(),
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        max_samples=None,
        base_estimator="deprecated",
    ):
        super().__init__(
            estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
            base_estimator=base_estimator,
        )

    def predict(self, X):
        """
        Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted values.
        """
        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        if self.n_outputs_ > 1:
            y_hat = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)
        else:
            y_hat = np.zeros((X.shape[0]), dtype=np.float64)

        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_accumulate_prediction)(e.predict, X, [y_hat], lock) for e in self.estimators_
        )

        y_hat /= len(self.estimators_)

        return y_hat

    @staticmethod
    def _get_oob_predictions(tree, X):
        """Compute the OOB predictions for an individual tree.

        Parameters
        ----------
        tree : DecisionTreeRegressor object
            A single decision tree regressor.
        X : ndarray of shape (n_samples, n_features)
            The OOB samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples, 1, n_outputs)
            The OOB associated predictions.
        """
        y_pred = tree.predict(X, check_input=False)
        if y_pred.ndim == 1:
            # single output regression
            y_pred = y_pred[:, np.newaxis, np.newaxis]
        else:
            # multioutput regression
            y_pred = y_pred[:, np.newaxis, :]
        return y_pred

    def _set_oob_score_and_attributes(self, X, y, scoring_function=None):
        """Compute and set the OOB score and attributes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.
        y : ndarray of shape (n_samples, n_outputs)
            The target matrix.
        scoring_function : callable, default=None
            Scoring function for OOB score. Defaults to `r2_score`.
        """
        self.oob_prediction_ = super()._compute_oob_predictions(X, y).squeeze(axis=1)
        if self.oob_prediction_.shape[-1] == 1:
            # drop the n_outputs axis if there is a single output
            self.oob_prediction_ = self.oob_prediction_.squeeze(axis=-1)

        if scoring_function is None:
            scoring_function = r2_score

        self.oob_score_ = scoring_function(y, self.oob_prediction_)

    def _compute_partial_dependence_recursion(self, grid, target_features):
        """Fast partial dependence computation.

        Parameters
        ----------
        grid : ndarray of shape (n_samples, n_target_features)
            The grid points on which the partial dependence should be
            evaluated.
        target_features : ndarray of shape (n_target_features)
            The set of target features for which the partial dependence
            should be evaluated.

        Returns
        -------
        averaged_predictions : ndarray of shape (n_samples,)
            The value of the partial dependence function on each grid point.
        """
        grid = np.asarray(grid, dtype=DTYPE, order="C")
        averaged_predictions = np.zeros(shape=grid.shape[0], dtype=np.float64, order="C")

        for tree in self.estimators_:
            # Note: we don't sum in parallel because the GIL isn't released in
            # the fast method.
            tree.tree_.compute_partial_dependence(grid, target_features, averaged_predictions)
        # Average over the forest
        averaged_predictions /= len(self.estimators_)

        return averaged_predictions

    def _more_tags(self):
        return {"multilabel": True}


# XXX: figure out how to make this either:
# - DML tree
# - DR tree
# - OrthoDML/DR tree
class CausalForest(ForestRegressor):
    _parameter_constraints: dict = {
        **ForestRegressor._parameter_constraints,
        **DecisionTreeRegressor._parameter_constraints,
    }
    _parameter_constraints.pop("splitter")

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
        honest_ratio=0.5,
    ):
        super().__init__(
            estimator=CausalTree(),
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "random_state",
                "ccp_alpha",
                "honest_ratio",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.honest_ratio = honest_ratio
