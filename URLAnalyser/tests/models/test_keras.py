import os
import mock
import tempfile

from URLAnalyser.models.keras import load_model
from URLAnalyser.models.keras import save_model

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)


def test_load_model(keras_model):
    with tempfile.TemporaryDirectory() as tmp_dir:
        file = os.path.join(tmp_dir, 'keras_model')

        keras_model.save(file)
        with mock.patch('URLAnalyser.models.keras.MODEL_DATA_DIRECTORY', tmp_dir):
            model = load_model('keras_model')

        assert isinstance(model, type(keras_model))


def test_save_model(keras_model):
    with tempfile.TemporaryDirectory() as tmp_dir:
        file = os.path.join(tmp_dir, 'keras_model')

        with mock.patch('URLAnalyser.models.keras.MODEL_DATA_DIRECTORY', tmp_dir):
            save_model(keras_model, 'keras_model')
        model = tf.keras.models.load_model(file)

        assert "keras_model" in os.listdir(tmp_dir)
        assert isinstance(model, type(keras_model))
