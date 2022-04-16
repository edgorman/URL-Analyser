import mock
import pytest
import pandas as pd

from URLAnalyser.features.lexical import get_lexical
from URLAnalyser.features.host import get_host
from URLAnalyser.features.content import get_content
from URLAnalyser.features.features import _get_method
from URLAnalyser.features.features import get_url_features


@pytest.mark.parametrize("dataset_name,expected", [
    ("lexical", get_lexical),
    ("host", get_host),
    ("content", get_content),
    ("", None)
])
def test_load_method(dataset_name, expected):
    assert _get_method(dataset_name) == expected


@pytest.mark.parametrize("url_data,dataset_name,feature_index,expected", [
    ({"name": ["example.com"]}, "lexical", 1, {"urlLength": [11]})
])
def test_get_url_features(url_data, dataset_name, feature_index, expected, features_data_directory):
    url_data = pd.DataFrame(data=url_data)
    expected = pd.DataFrame(data=expected)

    with mock.patch('URLAnalyser.utils.FEATURES_DATA_DIRECTORY', features_data_directory):
        features = get_url_features(url_data, dataset_name, feature_index)

    assert isinstance(features, pd.DataFrame)
    assert all([col in features.columns for col in expected.columns])
