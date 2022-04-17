import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score


def generate_predictions(model: object, x_test: pd.DataFrame, is_keras: bool) -> list:
    predictions = model.predict(x_test)
    if is_keras:
        predictions = (predictions > 0.5).astype(np.int)
    return predictions


def calculate_metrics(predictions: list, true_values: list) -> dict:
    return {
        'accuracy': round(accuracy_score(true_values, predictions), 3),
        'precision': round(precision_score(true_values, predictions), 3),
        'recall': round(recall_score(true_values, predictions), 3),
        'f1': round(f1_score(true_values, predictions), 3),
    }
