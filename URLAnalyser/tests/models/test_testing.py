import pytest

from URLAnalyser.models.testing import generate_predictions
from URLAnalyser.models.testing import calculate_metrics

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)


@pytest.mark.parametrize("is_keras", [
    (False),
    (True)
])
def test_generate_predictions(is_keras, sklearn_model, keras_model, train_test_data):
    _, x_test, _, _ = train_test_data
    model = keras_model if is_keras else sklearn_model
    result = generate_predictions(model, x_test, is_keras)

    assert len(result) == len(x_test)


@pytest.mark.parametrize("predictions,true_values,accuracy,precision,recall,f1", [
    (
        [0, 1, 1, 1, 0],
        [0, 1, 1, 0, 1],
        0.6,
        0.667,
        0.667,
        0.667
    )
])
def test_calculate_metrics(predictions, true_values, accuracy, precision, recall, f1):
    metrics = calculate_metrics(predictions, true_values)

    assert all(col in metrics for col in ['accuracy', 'precision', 'recall', 'f1'])
    assert metrics['accuracy'] == pytest.approx(accuracy, 0.1)
    assert metrics['precision'] == pytest.approx(precision, 0.1)
    assert metrics['recall'] == pytest.approx(recall, 0.1)
    assert metrics['f1'] == pytest.approx(f1, 0.1)
