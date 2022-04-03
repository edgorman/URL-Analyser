import pytest
import pandas as pd
from URLAnalyser.features.host import get_host


@pytest.mark.parametrize("urls,index,vocab,expected", [
    ({'serverCount': 1}, '2', {}, 1),
])
def test_get_host(urls, index, vocab, expected):
    features = get_host(urls, index, vocab)
    assert type(features) is pd.DataFrame
    assert len(features.columns) == expected
