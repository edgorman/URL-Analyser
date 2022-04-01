import os
import pickle


def save_sklearn_model(model, filename, path=os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "models")):
    pickle.dump(model, open(os.path.join(path, filename + ".pkl"), 'wb'))

def load_sklearn_model(filename, path=os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "models")):
    return pickle.load(open(os.path.join(path, filename + ".pkl"), 'rb'))