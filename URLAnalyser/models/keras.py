import os
import tensorflow as tf

from URLAnalyser.constants import MODEL_DATA_DIRECTORY


def load_model(filename):
    return tf.keras.models.load_model(os.path.join(MODEL_DATA_DIRECTORY, filename))


def save_model(model, filename):
    model.save(os.path.join(MODEL_DATA_DIRECTORY, filename))
