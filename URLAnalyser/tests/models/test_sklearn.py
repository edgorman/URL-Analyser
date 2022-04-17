import os
import mock
import pickle
import tempfile

from URLAnalyser.models.sklearn import save_model
from URLAnalyser.models.sklearn import load_model


def test_load_model(sklearn_model):
    with tempfile.TemporaryDirectory() as tmp_dir:
        pickle.dump(sklearn_model, open(os.path.join(tmp_dir, "sklearn_model.pkl"), 'wb'))
        with mock.patch('URLAnalyser.models.sklearn.MODEL_DATA_DIRECTORY', tmp_dir):
            model = load_model('sklearn_model')

        assert isinstance(model, type(sklearn_model))


def test_save_model(sklearn_model):
    with tempfile.TemporaryDirectory() as tmp_dir:
        file = os.path.join(tmp_dir, 'sklearn_model')

        with mock.patch('URLAnalyser.models.sklearn.MODEL_DATA_DIRECTORY', tmp_dir):
            save_model(sklearn_model, 'sklearn_model')
        model = pickle.load(open(file + '.pkl', 'rb'))

        assert "sklearn_model.pkl" in os.listdir(tmp_dir)
        assert isinstance(model, type(sklearn_model))
