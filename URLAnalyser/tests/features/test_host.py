import pytest
import pandas as pd

from URLAnalyser.features.host import get_host


@pytest.mark.parametrize("urls,index,expected", [
    ({'server_count': 1}, 2, 1),
])
def test_get_host(urls, index, expected):
    features = get_host(urls, index)
    assert isinstance(features, pd.DataFrame)
    assert len(features.columns) == expected
