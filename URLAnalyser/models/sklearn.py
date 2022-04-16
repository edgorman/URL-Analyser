import os
import pickle

from URLAnalyser.constants import MODEL_DATA_DIRECTORY


def save_model(model, filename):
    pickle.dump(model, open(os.path.join(MODEL_DATA_DIRECTORY, filename + ".pkl"), 'wb'))


def load_model(filename):
    return pickle.load(open(os.path.join(MODEL_DATA_DIRECTORY, filename + ".pkl"), 'rb'))
