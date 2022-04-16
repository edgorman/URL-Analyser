import os
import pickle

from URLAnalyser import app


def save_model(model, filename):
    pickle.dump(model, open(os.path.join(app.DATA_DIRECTORY, "models", filename + ".pkl"), 'wb'))


def load_model(filename):
    return pickle.load(open(os.path.join(app.DATA_DIRECTORY, "models", filename + ".pkl"), 'rb'))
