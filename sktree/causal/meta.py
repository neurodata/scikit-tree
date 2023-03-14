import numbers
from itertools import product

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone, is_classifier
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, check_cv
from sklearn.utils import check_random_state
from sklearn.utils.validation import _check_fit_params, check_is_fitted, indexable


def _parallel_crossfit_nuisance(
    estimator_treatment,
    estimator_outcome,
    X,
    y,
    t,
    sample_weight,
    train,
    test,
    verbose,
    split_progress=None,  # (split_idx, n_splits),
):
    """Crossfitting nuisance functions in parallel.

    Parameters
    ----------
    estimator_treatment : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.
    estimator_outcome : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.
    X : array-like of shape (n_samples, n_features)
        The covariate data to fit.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        The outcome variable.
    t : array-like of shape (n_samples,) or (n_samples, n_treatment_dims) or None
        The treatment variable.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights. If None, then samples are equally weighted. Splits
        that would create child nodes with net zero or negative weight are
        ignored while searching for a split in each node. In the case of
        classification, splits are also ignored if they would result in any
        single class carrying a negative weight in either child node.
    train : array-like of shape (n_train_samples,)
        Indices of training samples.
    test : array-like of shape (n_test_samples,)
        Indices of test samples.
    verbose : int
        The verbosity level.
    split_progress : {list, tuple} of int, default=None
        A list or tuple of format (<current_split_id>, <total_num_of_splits>).

    Returns
    -------
    result : dict with the following attributes
        fit_time : float
            Time spent for fitting in seconds.
        score_time : float
            Time spent for scoring in seconds.
        estimator_outcome : estimator object
            The fitted estimator.
        estimator_treatment : estimator object
            The fitted estimator.
        fit_error : str or None
            Traceback str if the fit failed, None if the fit succeeded.
        test_idx : array-like of shape (n_test_samples,)
            The indices of the test samples.
        y_residuals : array-like of shape (n_test_samples,)
            The residuals of ``y_test - estimator_outcome.predict(X_test)``.
        t_residuals : array-like of shape (n_test_samples,)
            The residuals of ``t_test - estimator_treatment.predict(X_test)``.
    """
    progress_msg = ""
    if verbose > 2:
        if split_progress is not None:
            progress_msg = f" {split_progress[0]+1}/{split_progress[1]}"

    if verbose > 1:
        params_msg = ""

    if verbose > 9:
        start_msg = f"[CV{progress_msg}] START {params_msg}"
        print(f"{start_msg}{(80 - len(start_msg)) * '.'}")

    # fit the nuisance functions on the
    X_train, X_test = X[train, :], X[test, :]
    y_train, y_test = y[train, :], y[test, :]
    t_train, t_test = t[train, :], t[test, :]

    y_pred = estimator_outcome.fit(X_train, y_train, sample_weight=sample_weight).predict(X_test)
    t_pred = estimator_treatment.fit(X_train, t_train, sample_weight=sample_weight).predict(X_test)

    # compute the resulting residuals
    y_residuals = y_test - y_pred
    t_residuals = t_test - t_pred

    result = dict()
    result["estimator_outcome"] = estimator_outcome
    result["estimator_treatment"] = estimator_treatment
    result["y_residuals"] = y_residuals
    result["t_residuals"] = t_residuals

    return result


def check_outcome_estimator(estimator_outcome, y, random_state):
    """Check outcome estimator and error-check.

    Parameters
    ----------
    estimator_outcome : estimator object
        The outcome estimator model. If None, then LassoCV will be used.
    y : array-like of shape (n_samples, n_output)
        The outcomes array.
    random_state : int
        Random seed.

    Returns
    -------
    estimator_outcome : estimator object
        The cloned outcome estimator model.
    """
    if estimator_outcome is None:
        return LassoCV(random_state=random_state)
    else:
        if not hasattr(estimator_outcome, "fit") or not hasattr(estimator_outcome, "predict"):
            raise RuntimeError(
                "All outcome estimator models must have the `fit` and `predict` functions implemented."
            )

        estimator_outcome = clone(estimator_outcome)
    # XXX: run some checks on y being compatible with the estimator
    if is_classifier(estimator_outcome) and not issubclass(y.type, numbers.Integral):
        raise RuntimeError(
            "Treatment array is not integers, but the treatment estimator is classification based. "
            "If treatment array is discrete, then treatment estimator should be a classifier. "
            "If treatment array is continuous, thent treatment estimator should be a regressor."
        )
    return estimator_outcome


def check_treatment_estimator(estimator_treatment, t, random_state):
    """Check treatment estimator and error-check.

    Parameters
    ----------
    estimator_treatment : estimator object
        The treatment estimator model. If None, then LogisticRegressionCV will be used.
    t : array-like of shape (n_samples, n_treatment_dims)
        The treatment array.
    random_state : int
        Random seed.

    Returns
    -------
    estimator_treatment : estimator object
        The cloned treatment estimator model.
    """
    if estimator_treatment is None:
        estimator_treatment = LogisticRegressionCV(random_state=random_state)
    else:
        if not hasattr(estimator_treatment, "fit") or not hasattr(estimator_treatment, "predict"):
            raise RuntimeError(
                "All treatment estimator models must have the `fit` and `predict` functions implemented."
            )

        estimator_treatment = clone(estimator_treatment)

    # XXX: run some checks on t being compatible with the estimator. i.e. discrete -> classifier, otw regressor
    if is_classifier(estimator_treatment) and not issubclass(t.type, numbers.Integral):
        raise RuntimeError(
            "Treatment array is not integers, but the treatment estimator is classification based. "
            "If treatment array is discrete, then treatment estimator should be a classifier. "
            "If treatment array is continuous, thent treatment estimator should be a regressor."
        )

    return estimator_treatment


class DML(MetaEstimatorMixin, BaseEstimator):
    """A meta-estimator for performining double machine-learning.

    Parameters
    ----------
    estimator : estimator object
        This is assumed to implement the scikit-learn estimator interface.
        Estimator needs to provide a ``fit``, and ``predict_prob`` function.
    estimator_outcome : estimator object, optional
        The estimator model for the outcome, by default :class:`sklearn.linear_model.LassoCV`.
        This is assumed to implement the scikit-learn estimator interface. Estimator needs
        to provide a ``fit``, and ``predict_prob`` function. For more details, see Notes.
    estimator_treatment : estimator object, optional
        The estimator model for the treatment, by default :class:`sklearn.linear_model.LogisticRegressionCV`.
        This is assumed to implement the scikit-learn estimator interface. Estimator needs to
        provide a ``fit``, and ``predict_prob`` function. For more details, see Notes.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        Pass an int for reproducible output across multiple
        function calls.
        See :term:`Glossary <random_state>`.
    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the different training and test sets.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.

    Attributes
    ----------
    estimator_outcome_ : list of estimator object
        Fitted nuisance estimators of the outcome.
    estimator_treatment_ : list of estimator object
        Fitted nuisance estimators of the treatment.
    estimator_final_ : estimator object
        Fitted final-stage estimator.
    
    Notes
    -----
    The CATE is modeled using the following equation:

    .. math ::
        Y - \\E[Y | X, W] = \\Theta(X) \\cdot (T - \\E[T | X, W]) + \\epsilon

    where ``Y`` are the outcomes, ``X`` are the observed covariates that confound treatment, ``T``
    and outcomes. ``W`` are observed covariates that act as controls. :math:`\\epsilon`` is the noise
    term of the data-generating model. The CATE is defined as :math:`Theta(X)`. 

    Double machine learning (DML) follows a two stage process, where a set of nuisance functions
    are estimated in the first stage in a crossfitting manner and a final stage estimates the CATE
    model :footcite:`chernozhukov2018double`. See the User-guide for a description of two-stage learners, specifically on orthogonal
    learning and doubly-robust learning. The two-stage procedure of DML follows the two steps:

    **1. Estimate nuisance functions**
    There are two nuisance functions that need to be estimated, :math:`q(X, W) = E[Y | X, W]` and
    :math:`f(X, W) = E[T | X, W]`, which are the models for the outcome and treatment
    respectively. Here, one needs to run a regression (or classification model if treatments/outcomes
    are discrete), that predicts outcomes and treatments given the covariates and control features.

    **2. Final stage regression of residuals on residuals**
    Once :math:`q(X, W)` and :math:`f(X, W)` are estimated in the first stage, we compute
    the residuals of the models with their observed outcome and treatment samples.

    .. math ::
        \\tilde{Y} = Y - q(X, W)
        \\tilde{T} = T - f(X, W)

    and then we fit a final stage model that takes the covariates, :math:`X` and :math:`\\tilde{T}`. That is
    it uses the residuals of the treatment model and the covariates to predict the residuals of the outcome.
    The fitted model represents the CATE.

    .. math ::
        \\hat{\\theta} = \\arg\\min_{\\Theta}\
        \\E_n\\left[ (\\tilde{Y} - \\Theta(X) \\cdot \\tilde{T})^2 \\right]

    In the DML process, ``T`` and ``Y`` can be either discrete, or continuous and the resulting process will
    be valid. For each stage of the DML process, one can use arbitrary sklearn-like estimators.

    **Estimation procedure with cross-fitting and cross-validation**

    In order to prevent overfitting to training data, the nuisance functions are estimated using training data
    and the nuisance values (i.e. the residuals) are computed using the testing data. This is done on K-folds of
    data, so the final residuals are computed from multiple model instantiations. If there is no cross-fitting,
    then the nuisance functions are estimated and evaluated on the entire dataset. If there is at least K-fold
    cross-fitting, then the data is split into K distinct groups and for each group, the model is trained on
    the rest of the data, and evaluated on the held-out test group.

    The final model then takes as input the residuals of outcome, residuals of treatment and covariates to
    estimate the CATE.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        estimator,
        estimator_outcome=None,
        estimator_treatment=None,
        cv=None,
        random_state=None,
        n_jobs=None,
        verbose=False,
    ) -> None:
        super().__init__()
        self.estimator = estimator
        self.estimator_outcome = estimator_outcome
        self.estimator_treatment = estimator_treatment
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y, *, t=None, groups=None, **fit_params):
        self._validate_params()
        random_state = check_random_state(self.random_state)

        # run checks on input data
        X, y = self._validate_data(X, y, multi_output=True, accept_sparse=False)
        self._validate_data(y=t, multi_output=True, accept_sparse=False)
        X, y, t, groups = indexable(X, y, t, groups)
        fit_params = _check_fit_params(X, fit_params)
        sample_weight = fit_params.get("sample_weight", None)

        # Determine output settings
        _, self.n_features_in_ = X.shape

        # get cross-validation object
        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        n_splits = cv.get_n_splits(X, y, groups)

        # XXX: this should not be required. If the user wants, they can pass in
        # a CV object
        if cv != self.cv and isinstance(cv, (KFold, StratifiedKFold)):
            cv.shuffle = True
            cv.random_state = random_state

        final_stage_estimator = clone(self.estimator)

        # initialize the models for treatment and outcome
        estimator_outcome = check_outcome_estimator(
            estimator_outcome=self.estimator_outcome, y=y, random_state=random_state
        )
        estimator_treatment = check_treatment_estimator(
            estimator_treatment=self.estimator_treatment, t=t, random_state=random_state
        )

        # fit the nuisance functions in parallel
        parallel = Parallel(n_jobs=self.n_jobs)
        with parallel:
            out = parallel(
                delayed(_parallel_crossfit_nuisance)(
                    estimator_treatment,
                    estimator_outcome,
                    final_stage_estimator,
                    X=X,
                    y=y,
                    t=t,
                    sample_weight=sample_weight,
                    train=train,
                    test=test,
                    verbose=self.verbose,
                    split_progress=(split_idx, n_splits),
                )
                for (split_idx, (train, test)) in enumerate(cv.split(X, y, groups))
            )

            y_residuals = np.zeros(y.shape)
            t_residuals = np.zeros(t.shape)

            for result in out:
                test = result["test_idx"]
                y_residuals[test, :] = result["y_residuals"]
                t_residuals[test, :] = result["t_residuals"]

        # now fit the final stage estimator
        final_stage_estimator.fit(X=X, y=y_residuals, t=t_residuals, **fit_params)

    def predict(self, X):
        """Predict the conditional average treatment effect given covariates X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        """
        pass

    def conditional_effect(self, X, t=None):
        """Compute conditional effect of

        Parameters
        ----------
        X : _type_
            _description_
        t : _type_, optional
            _description_, by default None
        """
        pass
