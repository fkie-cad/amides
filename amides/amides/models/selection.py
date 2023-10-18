"""This module contains functions and classes that aim to help with model selection."""
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.utils.validation import check_is_fitted

from amides.data import DataBunch, TrainTestSplit
from amides.utils import execution_time, get_logger


_logger = get_logger(__name__)


class HyperParameterOptimizer:
    """HyperParameterOptimizer to find optimal hyper parameters for given
    estimator and parameter search space.
    """

    def __init__(
        self,
        estimator,
        param_grid,
        search_method=GridSearchCV,
        cv_schema=None,
        n_jobs=4,
        scoring=None,
    ):
        """Init parameter optimizer.

        Parameters
        ----------
        estimator: sklearn.base.BaseEstimator
            The estimator which should be optimised.
        param_grid: dict
            The parameter grid used as search space.
        search_method: sklearn.model_selection.BaseSearchCV
            Hyper parameter search class instance.
        cv_schema: Optional[int]
            The cross-validation schema.
        scoring: str or callable
            The scoring function.
        """
        self._search_method = search_method(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv_schema,
            n_jobs=n_jobs,
            refit=True,
            pre_dispatch="1*n_jobs",
            verbose=1,
        )

    @property
    def best_parameters(self):
        check_is_fitted(self._search_method)
        return self._search_method.best_params_

    @property
    def best_estimator(self):
        check_is_fitted(self._search_method)
        return self._search_method.best_estimator_

    @property
    def best_score(self):
        check_is_fitted(self._search_method)
        return self._search_method.best_score_

    @execution_time
    def search_best_parameters(self, data):
        """Search best hyper parameters for configured estimator and
        search space using the given training data.

        Parameters
        ----------
        data: DataBunch
            Data used for searching given parameter search space.

        Raises
        ------
        ValueError
            In case provided data is not of type 'DataBunch'

        """
        if not isinstance(data, DataBunch):
            raise ValueError("data is not of required type DataBunch")

        _logger.info(
            "Searching parameters for: estimator=%s, search_method=%s, "
            "param_grid=%s, cv=%s",
            self._search_method.estimator.__class__.__name__,
            self._search_method.__class__.__name__,
            str(self._search_method.param_grid),
            str(self._search_method.cv),
        )

        self._search_method.fit(data.samples, data.labels)

        _logger.info("Best parameters: %s", self._search_method.best_params_)

    def predict(self, data):
        """Calculate predict values using the trained and optimised estimator.

        Parameters
        ----------
        data: DataBunch
            Data used to compute prediction values..

        Returns
        -------
        predict: np.ndarray
            The classification results on test_data using the
            optimised estimator.

        Raises
        ------
        ValueError
            If provided test data is not of type DataBunch.

        """
        if not isinstance(data, DataBunch):
            raise ValueError("data is not of required type DataBunch")

        _logger.debug(
            "Calculating prediction on: estimator=%s best_parameters=%s",
            self._search_method.best_estimator_.__class__.__name__,
            self._search_method.best_params_,
        )

        return self._search_method.best_estimator_.predict(data.samples)

    def search_and_predict(self, train_test_split):
        """Train and optimise estimator using training data and perform
        classification on test data afterwards.

        Parameters
        ----------
        train_test_split: TrainTestSplit
            Data used for parameter optimization and validation..

        Returns
        -------
        predict: np.ndarray
            The classification results of the best estimator on
            the validation data.

        Raises
        ------
        ValueError
            If provided data is not of type DataSplit.

        """
        if not isinstance(train_test_split, TrainTestSplit):
            raise ValueError(
                "data is of type {0}, but not of required"
                "type TrainTestSplit".format(type(train_test_split))
            )

        self.search_best_parameters(train_test_split.train_data)

        return self.predict(train_test_split.test_data)


class GridSearch:
    """GridSearch class to enable grid search without cross validation."""

    def __init__(
        self,
        estimator,
        param_grid,
        cv=1,
        scoring=None,
        refit=True,
        n_jobs=1,
        pre_dispatch="1*n_jobs",
        verbose=None,
    ):
        self._estimator = estimator
        self._param_grid = ParameterGrid(param_grid)
        self._cv = cv
        self._scoring = scoring
        self._refit = refit
        self._n_jobs = n_jobs
        self._pre_dispatch = pre_dispatch

        self.best_params_ = None
        self.best_estimator_ = None
        self.best_score_ = None

    @property
    def estimator(self):
        return self._estimator

    @property
    def cv(self):
        return self._cv

    @property
    def param_grid(self):
        return self._param_grid

    def fit(self, samples, labels):
        for params in self._param_grid:
            self._estimator.set_params(**params)
            self._estimator.fit(samples, labels)

            score = self._calculate_score(samples, labels)

            if self.best_score_ is None or score > self.best_score_:
                self.best_score_ = score
                self.best_params_ = self._estimator.get_params()

        self.best_estimator_ = self._estimator
        self.best_estimator_.set_params(**self.best_params_)
        self.best_estimator_.fit(samples, labels)

        if self._refit:
            self._estimator = self.best_estimator_

    def _calculate_score(self, samples, labels):
        if self._scoring is None:
            return self._estimator.score(samples, labels)
        else:
            return self._scoring(self._estimator, samples, labels)
