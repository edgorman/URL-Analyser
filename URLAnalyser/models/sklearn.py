import os
import pickle

from URLAnalyser.constants import DATA_DIRECTORY


def save_model(model, filename):
    pickle.dump(model, open(os.path.join(DATA_DIRECTORY, "models", filename + ".pkl"), 'wb'))


def load_model(filename):
    return pickle.load(open(os.path.join(DATA_DIRECTORY, "models", filename + ".pkl"), 'rb'))
