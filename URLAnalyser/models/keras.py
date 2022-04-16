import os
from keras.models import load_model as load

from URLAnalyser.constants import MODEL_DATA_DIRECTORY


def save_model(model, filename):
    model.save(os.path.join(MODEL_DATA_DIRECTORY, filename))


def load_model(filename):
    return load(os.path.join(MODEL_DATA_DIRECTORY, filename))
