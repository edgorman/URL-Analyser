import mock
import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from URLAnalyser.data.data import _load_lexical
from URLAnalyser.data.data import _load_host
from URLAnalyser.data.data import _load_content
from URLAnalyser.data.data import _load_method
from URLAnalyser.data.data import get_url_data
from URLAnalyser.data.data import get_train_test_data


@pytest.mark.parametrize("use_cache", [
    (False),
    (True)
])
def test_load_lexical(use_cache, url_data_directory):
    with mock.patch('URLAnalyser.data.data.URL_DATA_DIRECTORY', url_data_directory):
        result = _load_lexical(1, use_cache, False)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


@pytest.mark.parametrize("use_cache", [
    (False),
    (True)
])
def test_load_host(use_cache, url_data_directory):
    with mock.patch('URLAnalyser.data.data.URL_DATA_DIRECTORY', url_data_directory):
        result = _load_host(1, use_cache, False)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


@pytest.mark.parametrize("use_cache", [
    (False),
    (True)
])
def test_load_content(use_cache, url_data_directory):
    with mock.patch('URLAnalyser.data.data.URL_DATA_DIRECTORY', url_data_directory):
        result = _load_content(1, use_cache, False)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


@pytest.mark.parametrize("dataset_name,expected", [
    ("lexical", _load_lexical),
    ("host", _load_host),
    ("content", _load_content),
    ("", None)
])
def test_load_method(dataset_name, expected):
    assert _load_method(dataset_name) == expected


@pytest.mark.parametrize("dataset_name,ignore_columns,expected", [
    (
        "lexical",
        [],
        {
            'name': ["google.com", "bbc.co.uk"],
            'class': [0, 1]
        }
    ),
    (
        "host",
        ["registrar", "updated_date", "expiration_date", "speed", "latency"],
        {
            'name': ["google.com", "bbc.co.uk"],
            'class': [0, 1],
            'registrar': ["MarkMonitor Inc.", "British Broadcasting Corporation [Tag = BBC]"],
            'location': [None, None],
            'server_count': [4, 0],
            'creation_date': [datetime(1997, 9, 15, 4, 0, 0), np.datetime64('NaT')],
            'updated_date': [None, None],
            'expiration_date': [None, None],
            'speed': [None, None],
            'latency': [None, None],
        }
    ),
    (
        "content",
        ['content'],
        {
            'name': ["google.com", "bbc.co.uk"],
            'class': [0, 1],
            'type': ["text/html; charset=ISO-8859-1", "text/html"],
            'is_redirect': [False, False],
            'content': [None, None]
        }
    )
])
def test_get_url_data(dataset_name, ignore_columns, expected, url_data_directory):
    expected = pd.DataFrame(data=expected)
    with mock.patch('URLAnalyser.data.data.URL_DATA_DIRECTORY', url_data_directory):
        df = get_url_data(dataset_name, use_cache=False, is_keras=True)

    assert isinstance(df, pd.DataFrame)
    assert all([col in df.columns for col in expected.columns])

    # Drop columns that we can't compare easily
    expected.drop(ignore_columns, axis=1, inplace=True)
    df.drop(ignore_columns, axis=1, inplace=True)

    # Sort and remove index for dataframes so we can compare dataframes
    expected = expected.sort_values(by=expected.columns.tolist()).reset_index(drop=True)
    df = df.sort_values(by=df.columns.tolist()).reset_index(drop=True)

    assert df.equals(expected)


def test_get_train_test_data():
    url_data = pd.DataFrame(data={
        'name': ["google.com", "bbc.co.uk"],
        'feature': [0.1, 0.2],
        'class': [0, 1],
    })

    x_train, x_test, y_train, y_test = get_train_test_data(url_data)

    assert isinstance(x_train, pd.DataFrame) and isinstance(x_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series) and isinstance(y_test, pd.Series)
    assert len(x_train) + len(x_test) == 2
    assert len(y_train) + len(y_test) == 2
    assert "feature" in x_train.columns and "feature" in x_test.columns
