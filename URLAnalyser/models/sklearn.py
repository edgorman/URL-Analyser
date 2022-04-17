import os
import pickle
import sklearn

from URLAnalyser.constants import MODEL_DATA_DIRECTORY


def load_model(filename: str) -> sklearn.base.BaseEstimator:
    return pickle.load(open(os.path.join(MODEL_DATA_DIRECTORY, filename + ".pkl"), 'rb'))


def save_model(model: sklearn.base.BaseEstimator, filename: str) -> None:
    pickle.dump(model, open(os.path.join(MODEL_DATA_DIRECTORY, filename + ".pkl"), 'wb'))
