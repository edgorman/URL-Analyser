import os


PARENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
DATA_DIRECTORY = os.path.join(PARENT_DIRECTORY, "data")
URL_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, "urls")
MODEL_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, "models")
FEATURES_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, "features")
FEATURES_DIRECTORY = os.path.join(PARENT_DIRECTORY, "features")
MODELS_DIRECTORY = os.path.join(PARENT_DIRECTORY, "models")
