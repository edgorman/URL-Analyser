import talos as ta
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from URLAnalyser.log import Log
from URLAnalyser.constants import TALOS_DATA_DIRECTORY
from URLAnalyser.models.keras import create_layers
from URLAnalyser.models.keras import get_input_dim


def _tune_sklearn(model: object, params: dict, features: pd.DataFrame, labels: list) -> object:
    tuner = RandomizedSearchCV(
        estimator=model,
        scoring=["f1", "recall"],
        refit="recall",
        param_distributions=params,
        cv=2, n_jobs=-1, verbose=Log.verboseness
    )
    tuner.fit(features, labels)

    return model.set_params(**tuner.best_params_)


def _tune_keras(model: object, params: dict, features: pd.DataFrame, labels: list) -> object:
    print(TALOS_DATA_DIRECTORY)
    tuner = ta.Scan(features, labels, params, model, TALOS_DATA_DIRECTORY)
    results = tuner.data[tuner.data.recall_m == tuner.data.recall_m.max()]

    best_params = {}
    for name, _ in params.items():
        best_params[name] = results.iloc[0][name]

    return create_layers(
        get_input_dim(model),
        (best_params['first_neuron'], best_params['first_activation']),
        best_params['dropout'],
        (best_params['second_neuron'], best_params['second_activation']),
    )


def tune_hyperparameters(model: object, is_keras: bool, params: dict, features: pd.DataFrame, labels: list) -> object:
    tune_method = _tune_keras if is_keras else _tune_sklearn
    return tune_method(model, params, features, labels)
