import pytest
import numpy as np

from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import NotFittedError

from amides.data import DataBunch, TrainTestSplit
from amides.models.selection import HyperParameterOptimizer


@pytest.fixture
def train_data():
    train_data = [
        "some-commandline",
        "some-other-commandline",
        "this-commandline",
        "that-commandline"
    ]
    c_vect = CountVectorizer()
    train_data_transformed = c_vect.fit_transform(train_data).toarray()

    return DataBunch(np.array(train_data_transformed), np.array([0, 1, 1, 0]), 
                     ["benign", "matching"])

@pytest.fixture
def train_test_split(train_data):
    test_data = [
        "more-commandline",
        "even-more-commandline",
        "wow-commandline",
        "such-commandline"
    ]
    train_data = [
        "some-commandline",
        "some-other-commandline",
        "this-commandline",
        "that-commandline"
    ]
    c_vect = CountVectorizer()
    train_data_transformed = c_vect.fit_transform(train_data).toarray()
    test_data_transformed = c_vect.transform(test_data).toarray()

    test_data = DataBunch(test_data_transformed, np.array([1, 0, 1, 0]), 
                          ["benign", "matching"])
    train_data = DataBunch(train_data_transformed, np.array([0, 1, 1, 0]), 
                          ["benign", "matching"])

    return TrainTestSplit(train_data, test_data, name="some_rule")

class TestHParamOptimizer:

    @pytest.fixture
    def hp_optimizer(self):
        return HyperParameterOptimizer(
            estimator=SVC(),
            param_grid={
                "C": np.logspace(-1, 1, num=1),
                "gamma": np.logspace(1, 5, num=1)
            },
            search_method=GridSearchCV,
            cv_schema=2)

    def test_init(self):
        optimizer = HyperParameterOptimizer(
            estimator=SVC(),
            param_grid={
                "C": np.logspace(-1, 1, num=1),
                "gamma": np.logspace(1, 5, num=1)
            },
            search_method=GridSearchCV,
            cv_schema=2)
        assert optimizer

    def test_init_invalid_param_grid(self):
        with pytest.raises((ValueError, AttributeError)):
            _ = HyperParameterOptimizer(
                estimator=SVC(),
                param_grid="invalid",
                search_method=GridSearchCV,
                cv_schema=2
            )

    def test_init_invalid_search_method(self):
        with pytest.raises(TypeError):
            _ = HyperParameterOptimizer(
                estimator=SVC(),
                param_grid={
                    "C": np.logspace(-1, 1, num=1),
                    "gamma": np.logspace(1, 5, num=1)
                },
                search_method="invalid",
                cv_schema=2
            )

    def test_best_parameters_not_fitted(self, hp_optimizer):    
        with pytest.raises(NotFittedError):
            _ = hp_optimizer.best_parameters

    def test_best_estimator_not_fitted(self, hp_optimizer):
        with pytest.raises(NotFittedError):
            _ = hp_optimizer.best_parameters

    def test_best_score_not_fitted(self, hp_optimizer):
        with pytest.raises(NotFittedError):
            _ = hp_optimizer.best_parameters

    def test_search_best_parameters(self, train_data):
        optimizer = HyperParameterOptimizer(
            estimator=SVC(),
            param_grid={
                "C": np.logspace(-1, 1, num=1),
                "gamma": np.logspace(1, 5, num=1)
            },
            search_method=GridSearchCV,
            cv_schema=2
        )

        optimizer.search_best_parameters(train_data)
        assert optimizer.best_estimator
        assert optimizer.best_parameters
        assert optimizer.best_score

    def test_search_best_parameters_invalid_cv_schema(self, train_data):
        optimizer = HyperParameterOptimizer(
                estimator=SVC(),
                param_grid={
                    "C": np.logspace(-1, 1, num=1),
                    "gamma": np.logspace(1, 5, num=1)
                },
                search_method=GridSearchCV,
                cv_schema="invalid"
            )

        with pytest.raises(ValueError):
            optimizer.search_best_parameters(train_data)

    def test_search_best_parameters_invalid_input(self, hp_optimizer):
        with pytest.raises(ValueError):
            hp_optimizer.search_best_parameters("invalid")

    def test_search_and_predict(self, hp_optimizer, train_test_split):
        predict = hp_optimizer.search_and_predict(train_test_split)
        assert len(predict) == 4

    def test_search_and_predict_invalid_input(self, hp_optimizer):
        with pytest.raises(ValueError):
            _ = hp_optimizer.search_and_predict(train_test_split)
