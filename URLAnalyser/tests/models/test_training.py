import mock
import pytest

from URLAnalyser.models.training import tune_hyperparameters


@pytest.mark.parametrize("is_keras,hyperparameters", [
    (
        False,
        {
            "C": [0.001, 0.01, 0.1, 1, 10],
            "kernel": ['linear', 'poly', 'rbf'],
            "gamma": ['scale', 'auto'],
            "degree": [1, 2, 3, 5, 10]
        }
    )
    # (
    #     True,
    #     {
    #         "first_activation": ['sigmoid', 'elu'],
    #         "second_activation": ['sigmoid', 'elu'],
    #         "dropout": [0.2, 0.5],
    #         "first_neuron": [128, 256],
    #         "second_neuron": [128, 256]
    #     }
    # )
])
def test_tune_hyperparameters(is_keras,
                              hyperparameters,
                              sklearn_model,
                              keras_model,
                              train_test_data,
                              talos_data_directory):

    x_train, _, y_train, _ = train_test_data
    model = keras_model if is_keras else sklearn_model

    with mock.patch('URLAnalyser.models.training.TALOS_DATA_DIRECTORY', talos_data_directory):
        tuned_model = tune_hyperparameters(model, is_keras, hyperparameters, x_train, y_train)

    assert isinstance(tuned_model, type(model))
