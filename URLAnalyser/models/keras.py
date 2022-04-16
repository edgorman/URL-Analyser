import os
from keras.models import load_model as load

from URLAnalyser import app


def save_model(model, filename):
    model.save(os.path.join(app.DATA_DIRECTORY, "models", filename))


def load_model(filename):
    return load(os.path.join(app.DATA_DIRECTORY, "models", filename))
