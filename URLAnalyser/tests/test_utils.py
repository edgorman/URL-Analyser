import os
import pytest
import pandas as pd
from sklearn.svm import SVC
from keras import Model

from URLAnalyser.utils import bag_of_words
from URLAnalyser.utils import generate_config_filename
from URLAnalyser.utils import load_json_as_dict
from URLAnalyser.utils import url_is_valid
from URLAnalyser.utils import config_is_valid
from URLAnalyser.utils import model_is_stored
from URLAnalyser.utils import get_class
from URLAnalyser.utils import safe_division


def test_load_json_as_dict(test_directory):
    filename = os.path.join(test_directory, "data", "models", "results-dict.json")
    loaded_dict = load_json_as_dict(filename)

    assert isinstance(loaded_dict, dict)
    assert "svm" in loaded_dict.keys()
    assert "sklearn.svm.SVC" == loaded_dict["svm"]["class"]


@pytest.mark.parametrize("url,expected", [
    ("google.com", True),
    ("thenamrog.com", False),
])
def test_url_is_valid(url, expected):
    assert url_is_valid(url) == expected


@pytest.mark.skip(reason="need to mock DATA_DIRECTORY constant to test directory")
@pytest.mark.parametrize("model_name,dataset_name,feature_index,expected", [
    ("svm", "lexical", "0", True),
    ("", "lexical", "0", False),
    ("svm", "", "0", False),
    ("svm", "lexical", "-1", False),
])
def test_config_is_valid(model_name, dataset_name, feature_index, expected):
    assert config_is_valid(model_name, dataset_name, feature_index) == expected


@pytest.mark.parametrize("model_name,dataset_name,feature_index,expected", [
    ("a", "b", "c", "a-b-c"),
])
def test_generate_config_filename(model_name, dataset_name, feature_index, expected):
    assert generate_config_filename(model_name, dataset_name, feature_index) == expected


@pytest.mark.skip(reason="need to mock DATA_DIRECTORY constant to test directory")
@pytest.mark.parametrize("model_name,dataset_name,feature_index,expected", [
    ("a", "b", "c", True),
    ("a", "b", "d", False),
])
def test_model_is_stored(model_name, dataset_name, feature_index, expected):
    filename = generate_config_filename(model_name, dataset_name, feature_index)
    assert model_is_stored(filename) == expected


@pytest.mark.parametrize("class_name,expected", [
    ("sklearn.svm.SVC", SVC),
    ("keras.Model", Model),
])
def test_get_class(class_name, expected):
    assert isinstance(get_class(class_name), type(expected))


@pytest.mark.skip(reason="need to mock DATA_DIRECTORY constant to test directory")
def test_bag_of_words():
    series = pd.Series([
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ])
    # TODO: constants.DATA_DIRECTORY to be test_directory
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
