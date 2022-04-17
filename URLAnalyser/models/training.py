import sklearn
from sklearn.model_selection import RandomizedSearchCV
import tensorflow as tf
import pandas as pd

from URLAnalyser.log import Log


def _tune_sklearn(model: sklearn.base.BaseEstimator, params: dict) -> sklearn.base.BaseEstimator:
    return RandomizedSearchCV(
        estimator=model,
        scoring=["f1", "recall"],
        refit="recall",
        param_distributions=params,
        cv=2, n_jobs=-1, verbose=Log.verboseness
    )


def _tune_keras(model: tf.keras.models.Model, params: dict) -> tf.keras.models.Model:
    return None


def tune_hyperparameters(model: object, is_keras: bool, params: dict, features: pd.DataFrame, labels: list) -> object:
    tuned_model = _tune_keras(model, params) if is_keras else _tune_sklearn(model, params)
    tuned_model.fit(features, labels)
    return model.set_params(**tuned_model.best_params_)
