import os
import pytest
import pandas as pd
from sklearn.svm import SVC
from keras import Model
from URLAnalyser.utils import bag_of_words, generate_model_filename, is_model_stored, load_json_as_dict, safe_division
from URLAnalyser.utils import is_valid_url
from URLAnalyser.utils import is_valid_model
from URLAnalyser.utils import is_model_stored
from URLAnalyser.utils import get_class
from URLAnalyser.utils import split_urls
from URLAnalyser.utils import safe_division


def test_load_json_as_dict(results_dict_path):
    loaded_dict = load_json_as_dict(results_dict_path)

    assert type(loaded_dict) is dict
    assert "svm" in loaded_dict.keys()
    assert "sklearn.svm.SVC" == loaded_dict["svm"]["class"]

@pytest.mark.skip(reason="Not yet implemented: utils.is_valid_url")
@pytest.mark.parametrize("url,expected", [
    ("https://google.com", True),
    ("https://thenamrog.com", False),
])
def test_is_valid_url(url, expected):
    assert is_valid_url(url) == expected

@pytest.mark.parametrize("model_name,dataset_name,feature_index,expected", [
    ("svm", "lexical", "0", True),
    ("", "lexical", "0", False),
    ("svm", "", "0", False),
    ("svm", "lexical", "-1", False),
])
def test_is_valid_model(model_name, dataset_name, feature_index, expected, results_dict_path):
    model_dict = load_json_as_dict(results_dict_path)

    assert is_valid_model(model_dict, model_name, dataset_name, feature_index) == expected

@pytest.mark.parametrize("model_name,dataset_name,feature_index,expected", [
    ("a", "b", "c", "a-b-c"),
])
def test_generate_model_filename(model_name, dataset_name, feature_index, expected):
    assert generate_model_filename(model_name, dataset_name, feature_index) == expected

@pytest.mark.parametrize("model_name,dataset_name,feature_index,expected", [
    ("a", "b", "c", True),
    ("a", "b", "d", False),
])
def test_is_model_stored(model_name, dataset_name, feature_index, expected):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "models")
    assert is_model_stored(model_name, dataset_name, feature_index, path) == expected

@pytest.mark.parametrize("class_name,expected", [
    ("sklearn.svm.SVC", SVC),
    ("keras.Model", Model),
])
def test_get_class(class_name, expected):
    assert type(get_class(class_name)) == type(expected)

def test_get_urls(url_dataframe):
    assert type(url_dataframe) is pd.DataFrame
    assert all([col in ["name", "class"] for col in url_dataframe.columns])

def test_split_urls(url_dataframe):
    train_x, test_x, train_y, test_y = split_urls(url_dataframe)
    assert len(train_x) + len(test_x) == 2
    assert len(train_y) + len(test_y) == 2

def test_bag_of_words():
    series = pd.Series([
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ])
    features = bag_of_words(pd.DataFrame(), series, [])
    assert list(features.columns) == ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
    assert len(features) == 4

@pytest.mark.parametrize("left,right,expected", [
    (5, 0, 0),
    (5, 1, 5),
])
def test_safe_division(left, right, expected):
    assert safe_division(left, right) == expected
