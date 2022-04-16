import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from URLAnalyser.data.data import load_url_data


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
        ["updated_date", "expiration_date", "speed", "latency"],
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
def test_load_url_data(dataset_name, ignore_columns, expected):
    # TODO: Update app.DATA_DIRECTORY to use test version of directory
    expected = pd.DataFrame(data=expected)
    df = load_url_data(dataset_name, use_cache=False, is_keras=True)

    assert isinstance(expected, pd.DataFrame)
    assert all([col in expected.columns for col in df.columns])

    # Drop columns that we can't compare easily
    expected.drop(ignore_columns, axis=1, inplace=True)
    df.drop(ignore_columns, axis=1, inplace=True)

    # Sort and remove index for dataframes so we can compare dataframes
    expected = expected.sort_values(by=expected.columns.tolist()).reset_index(drop=True)
    df = df.sort_values(by=df.columns.tolist()).reset_index(drop=True)

    assert df.equals(expected)
