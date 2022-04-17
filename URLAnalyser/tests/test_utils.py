import os
import mock
import pytest
import pickle
import tempfile
import pandas as pd
from sklearn.svm import SVC
from keras import Model

from URLAnalyser.utils import load_json_as_dict
from URLAnalyser.utils import save_dict_as_json
from URLAnalyser.utils import load_csv_as_dataframe
from URLAnalyser.utils import save_dataframe_to_csv
from URLAnalyser.utils import url_is_valid
from URLAnalyser.utils import generate_config_filename
from URLAnalyser.utils import config_is_valid
from URLAnalyser.utils import model_is_stored
from URLAnalyser.utils import get_class
from URLAnalyser.utils import safe_division
from URLAnalyser.utils import bag_of_words


def test_load_json_as_dict(model_data_directory):
    filename = os.path.join(model_data_directory, "results-dict.json")
    loaded_dict = load_json_as_dict(filename)

    assert isinstance(loaded_dict, dict)
    assert "svm" in loaded_dict.keys()
    assert "sklearn.svm.SVC" == loaded_dict["svm"]["class"]


def test_save_dict_as_json(tmpdir):
    tmp_file = tmpdir.join('test.json')
    save_dict_as_json({'test': 0}, tmp_file.strpath)

    assert tmp_file.read() == '{"test": 0}'


def test_load_csv_as_dataframe(url_data_directory):
    filename = os.path.join(url_data_directory, "whitelist.txt")
    loaded_df = load_csv_as_dataframe(filename)

    assert isinstance(loaded_df, pd.DataFrame)
    assert "name" in loaded_df.columns
    assert len(loaded_df) == 1


def test_save_dataframe_to_csv(tmpdir):
    tmp_file = tmpdir.join('test.csv')
    save_dataframe_to_csv(pd.DataFrame(data={"test": ["0"]}), tmp_file.strpath)

    assert tmp_file.read() == 'test\n0\n'


@pytest.mark.parametrize("url,expected", [
    ("google.com", True),
    ("thenamrog.com", False),
])
def test_url_is_valid(url, expected):
    assert url_is_valid(url) == expected


@pytest.mark.parametrize("model_name,dataset_name,feature_index,expected", [
    ("svm", "lexical", 0, True),
    ("", "lexical", 0, False),
    ("svm", "", 0, False),
    ("svm", "lexical", -1, False),
])
def test_config_is_valid(model_name, dataset_name, feature_index, expected, model_data_directory):
    with mock.patch('URLAnalyser.utils.MODEL_DATA_DIRECTORY', model_data_directory):
        assert config_is_valid(model_name, dataset_name, feature_index) == expected


@pytest.mark.parametrize("model_name,dataset_name,feature_index,expected", [
    ("a", "b", "c", "a-b-c"),
])
def test_generate_config_filename(model_name, dataset_name, feature_index, expected):
    assert generate_config_filename(model_name, dataset_name, feature_index) == expected


@pytest.mark.parametrize("model_name,dataset_name,feature_index,expected", [
    ("a", "b", "c", True),
    ("a", "b", "d", False),
])
def test_model_is_stored(model_name, dataset_name, feature_index, expected, sklearn_model):
    with tempfile.TemporaryDirectory() as tmp_dir:
        filename = generate_config_filename(model_name, dataset_name, feature_index)

        if expected:
            pickle.dump(sklearn_model, open(os.path.join(tmp_dir, filename), 'wb'))

        with mock.patch('URLAnalyser.utils.MODEL_DATA_DIRECTORY', tmp_dir):
            assert model_is_stored(filename) == expected


@pytest.mark.parametrize("class_name,expected", [
    ("sklearn.svm.SVC", SVC),
    ("keras.Model", Model),
])
def test_get_class(class_name, expected):
    assert isinstance(get_class(class_name), type(expected))


def test_bag_of_words(features_data_directory):
    series = pd.Series([
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ])

    with mock.patch('URLAnalyser.utils.FEATURES_DATA_DIRECTORY', features_data_directory):
        features = bag_of_words(pd.DataFrame(), series, 'example')

    assert len(features) == 4
    assert list(features.columns) == [
        'and',
        'document',
        'first',
        'is',
        'one',
        'second',
        'the',
        'third',
        'this'
    ]


@pytest.mark.parametrize("left,right,expected", [
    (5, 0, 0),
    (5, 1, 5),
])
def test_safe_division(left, right, expected):
    assert safe_division(left, right) == expected
