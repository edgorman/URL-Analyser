import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score


def generate_predictions(model, x_test, is_keras):
    predictions = model.predict(x_test)
    if is_keras:
        predictions = (predictions > 0.5).astype(np.int)
    return predictions


def calculate_metrics(predictions, true_values):
    return {
        'accuracy': round(accuracy_score(true_values, predictions), 3),
        'precision': round(precision_score(true_values, predictions), 3),
        'recall': round(recall_score(true_values, predictions), 3),
        'f1': round(f1_score(true_values, predictions), 3),
    }
